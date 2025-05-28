import os
import sys
import datetime
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp

import dataset
from model.get_model import get_model

# IOU损失函数
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

# 结构化损失函数
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# 计算MAE
def calculate_mae(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()

# 计算F-measure
def calculate_fmeasure(pred, gt, beta_squared=0.3):
    eps = 1e-8
    pred_bin = (pred >= 0.5).float()
    tp = (pred_bin * gt).sum()
    fp = pred_bin.sum() - tp
    fn = gt.sum() - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fmeasure = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall + eps)
    return fmeasure.item()

# 主训练函数
def train(Dataset, parser):
    args = parser.parse_args()
    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _LOSS_ = args.loss
    _SAVEPATH_ = args.savepath
    os.makedirs("result", exist_ok=True)

    cfg = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_, momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)
    data = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=6)
    net = get_model(cfg, _MODEL_)
    net.train(True).cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        (base if 'encoder' in name or 'network' in name else head).append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O2", keep_batchnorm_fp32=True)
    sw = SummaryWriter(cfg.savepath)

    loss_history = {'loss1': [], 'loss2': [], 'loss3': [], 'loss4': [], 'lossp': [], 'mae': [], 'fmeasure': []}
    global_step = 0

    for epoch in range(cfg.epoch):
        lr_factor = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1))
        optimizer.param_groups[0]['lr'] = lr_factor * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = lr_factor * cfg.lr

        total_mae = 0.0
        total_fm = 0.0
        total_samples = 0
        epoch_loss_sum = {'loss1': 0.0, 'loss2': 0.0, 'loss3': 0.0, 'loss4': 0.0, 'lossp': 0.0}
        total_batches = 0

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()
            out2, out3, out4, out5, pose = net(image)

            if _LOSS_ == "CPR":
                loss1 = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
                loss2 = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
                loss3 = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
                loss4 = F.binary_cross_entropy_with_logits(out5, mask) + iou_loss(out5, mask)
                lossp = F.binary_cross_entropy_with_logits(pose, mask) + iou_loss(pose, mask)
            elif _LOSS_ == "STR":
                loss1 = structure_loss(out2, mask)
                loss2 = structure_loss(out3, mask)
                loss3 = structure_loss(out4, mask)
                loss4 = structure_loss(out5, mask)
                lossp = structure_loss(pose, mask)

            total_loss = loss1 + loss2 + loss3 + loss4 + lossp
            optimizer.zero_grad()
            with amp.scale_loss(total_loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            # 计算指标
            pred = torch.sigmoid(pose)
            batch_mae = calculate_mae(pred, mask)
            batch_fm = calculate_fmeasure(pred, mask)
            total_mae += batch_mae * image.size(0)
            total_fm += batch_fm * image.size(0)
            total_samples += image.size(0)
            total_batches += 1

            for k, v in zip(['loss1', 'loss2', 'loss3', 'loss4', 'lossp'], [loss1, loss2, loss3, loss4, lossp]):
                epoch_loss_sum[k] += v.item()

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

            if step % 10 == 0:
                print(f"{datetime.datetime.now()} | step:{global_step}/{epoch + 1}/{cfg.epoch} | lr={optimizer.param_groups[0]['lr']:.6f} | " +
                      f"loss1={loss1.item():.6f} | loss2={loss2.item():.6f} | loss3={loss3.item():.6f} | loss4={loss4.item():.6f} | lossp={lossp.item():.6f}")

        # 记录每个epoch的平均loss
        for k in ['loss1', 'loss2', 'loss3', 'loss4', 'lossp']:
            avg_loss = epoch_loss_sum[k] / total_batches
            loss_history[k].append(avg_loss)
            sw.add_scalar(k, avg_loss, epoch + 1)

        # 记录epoch指标
        epoch_mae = total_mae / total_samples
        epoch_fm = total_fm / total_samples
        loss_history['mae'].append(epoch_mae)
        loss_history['fmeasure'].append(epoch_fm)
        sw.add_scalar('MAE', epoch_mae, epoch + 1)
        sw.add_scalar('F-measure', epoch_fm, epoch + 1)

        torch.save(net.state_dict(), f"{cfg.savepath}/{_MODEL_}{epoch + 1}")

    # ======== 绘图部分 ========
    epochs = range(1, cfg.epoch + 1)

    # 绘制Loss曲线
    plt.figure(figsize=(10, 6))
    for k in ['loss1', 'loss2', 'loss3', 'loss4', 'lossp']:
        plt.plot(epochs, loss_history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("result/loss_curves.png")
    plt.close()

    # 绘制MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history['mae'], 'b-', label='MAE')
    plt.xlabel("Epoch")
    plt.ylabel("MAE Value")
    plt.title("Mean Absolute Error Curve")
    plt.grid(True)
    plt.savefig("result/mae_curve.png")
    plt.close()

    # 绘制F-measure曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history['fmeasure'], 'r-', label='F-measure')
    plt.xlabel("Epoch")
    plt.ylabel("F-measure Value")
    plt.title("F-measure Curve")
    plt.grid(True)
    plt.savefig("result/fmeasure_curve.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ICON-R')
    parser.add_argument("--dataset", default='../data/HSaliency/train')
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=14)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--loss", default='CPR')
    parser.add_argument("--savepath", default='../checkpoint/ICON/ICON-R')
    parser.add_argument("--valid", type=bool, default=True)
    train(dataset, parser)
