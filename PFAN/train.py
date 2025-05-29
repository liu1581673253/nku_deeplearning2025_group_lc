"""
PyTorch Implementation of Pyramid Feature Attention Network for Saliency Detection
Complete Training Script with Metrics Visualization
"""

from __future__ import print_function, division, absolute_import
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Custom Modules
from src.dataloader import SODLoader
from src.model import SODModel
from src.loss import EdgeSaliencyLoss

def parse_arguments():
    """配置训练参数"""
    parser = argparse.ArgumentParser(description='Pyramid Feature Attention Network Trainer')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=391, help='Total training epochs')
    parser.add_argument('--bs', type=int, default=6, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0., help='Weight decay')
    
    # Data Parameters
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--aug', type=bool, default=True, help='Enable data augmentation')
    parser.add_argument('--n_worker', type=int, default=2, help='Data loader workers')
    
    # Model Management
    parser.add_argument('--test_interval', type=int, default=2, help='Test interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=None, 
                       help='Model saving interval (epochs)')
    parser.add_argument('--save_opt', type=bool, default=False, 
                       help='Save optimizer state')
    parser.add_argument('--base_save_path', type=str, default='./models',
                       help='Base path for model storage')
    
    # Loss Parameters
    parser.add_argument('--alpha_sal', type=float, default=0.7, 
                       help='Saliency loss weight')
    parser.add_argument('--wbce_w0', type=float, default=1.0, 
                       help='Weighted BCE w0 parameter')
    parser.add_argument('--wbce_w1', type=float, default=1.15,
                  help='Weighted BCE w1 parameter')
    
    return parser.parse_args()

class TrainingEngine:
    """训练引擎类"""
    def __init__(self, args):
        # 参数初始化
        self._init_params(args)
        self._setup_device()
        self._build_model()
        self._prepare_data()
        self._init_metrics_records()
        
    def _init_params(self, args):
        """初始化训练参数"""
        self.epochs = args.epochs
        self.bs = args.bs
        self.lr = args.lr
        self.wd = args.wd
        self.img_size = args.img_size
        self.aug = args.aug
        self.n_worker = args.n_worker
        self.test_interval = args.test_interval
        self.save_interval = args.save_interval
        self.save_opt = args.save_opt
        self.alpha_sal = args.alpha_sal
        self.wbce_params = (args.wbce_w0, args.wbce_w1)
        
        # 模型保存路径
        self.model_path = os.path.join(
            args.base_save_path,
            f"alph-{self.alpha_sal}_wbce_w0-{self.wbce_params[0]}_w1-{self.wbce_params[1]}"
        )
        self._create_dirs()
        
    def _setup_device(self):
        """配置计算设备"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self):
        """构建模型和优化器"""
        self.model = SODModel().to(self.device)
        self.criterion = EdgeSaliencyLoss(device=self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.wd
        )
        
    def _prepare_data(self):
        """准备数据加载器"""
        self.train_data = SODLoader(mode='train', augment_data=self.aug, target_size=self.img_size)
        self.test_data = SODLoader(mode='test', augment_data=False, target_size=self.img_size)
        
        self.train_loader = DataLoader(
            self.train_data, 
            batch_size=self.bs, 
            shuffle=True, 
            num_workers=self.n_worker
        )
        self.test_loader = DataLoader(
            self.test_data, 
            batch_size=self.bs, 
            shuffle=False, 
            num_workers=self.n_worker
        )
        
    def _create_dirs(self):
        """创建存储目录"""
        os.makedirs(os.path.join(self.model_path, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, 'optimizers'), exist_ok=True)
        os.makedirs(os.path.join(self.model_path, 'metrics'), exist_ok=True)
        
    def _init_metrics_records(self):
        """初始化指标记录器"""
        self.train_losses = []     # 记录每个epoch的训练损失
        self.val_epochs = []       # 记录进行验证的epoch编号
        self.test_losses = []      # 记录验证损失
        self.mae_scores = []       # 记录MAE
        self.f1_scores = []        # 记录F1分数
        
    def _compute_metrics(self, pred, gt):
        """计算评估指标"""
        tp = torch.sum((pred > 0.5) & (gt > 0.5)).float()
        fp = torch.sum((pred > 0.5) & (gt <= 0.5)).float()
        fn = torch.sum((pred <= 0.5) & (gt > 0.5)).float()
        
        # MAE计算
        mae = torch.mean(torch.abs(pred - gt)).item()
        
        # F1 Score计算
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        
        return mae, f1.item()
        
    def train(self):
        """主训练循环"""
        best_mae = float('inf')
        for epoch in range(self.epochs):
            # ================= 训练阶段 ================= #
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                pred_masks, reg_term = self.model(images)
                
                # 损失计算
                loss = self.criterion(pred_masks, masks) + reg_term
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            # 记录训练损失
            avg_train_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"[TRAIN] Epoch: {epoch+1}/{self.epochs} | Loss: {avg_train_loss:.4f}")
            
            # ================= 验证阶段 ================= #
            validate_flag = False
            if self.test_interval and (epoch % self.test_interval == 0):
                validate_flag = True
            if self.save_interval and (epoch % self.save_interval == 0):
                validate_flag = True
                
            if validate_flag:
                test_loss, mae, f1 = self._validate(epoch)
                self.val_epochs.append(epoch)
                self.test_losses.append(test_loss)
                self.mae_scores.append(mae)
                self.f1_scores.append(f1)
                
                # 保存最佳模型
                if mae < best_mae:
                    best_mae = mae
                    self._save_checkpoint(epoch, 'best', test_loss, mae, f1)
                    
                # 定期保存
                if self.save_interval and (epoch % self.save_interval == 0):
                    self._save_checkpoint(epoch, 'regular', test_loss, mae, f1)
                    
        # ================ 训练后处理 ================ #
        self._save_training_metrics()
        self._plot_metrics()
        print("Training completed!")
        
    def _validate(self, epoch):
        """验证阶段"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_f1 = 0.0
        
        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                pred_masks, reg_term = self.model(images)
                loss = self.criterion(pred_masks, masks) + reg_term
                mae, f1 = self._compute_metrics(pred_masks, masks)
                
                total_loss += loss.item()
                total_mae += mae
                total_f1 += f1
                
        avg_loss = total_loss / len(self.test_loader)
        avg_mae = total_mae / len(self.test_loader)
        avg_f1 = total_f1 / len(self.test_loader)
        
        print(f"[VAL] Epoch: {epoch+1} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | F1: {avg_f1:.4f}")
        return avg_loss, avg_mae, avg_f1
    
    def _save_checkpoint(self, epoch, ckpt_type, loss, mae, f1):
        """保存模型检查点"""
        state = {
            'epoch': epoch+1,
            'model_state': self.model.state_dict(),
            'loss': loss,
            'mae': mae,
            'f1': f1
        }
        
        filename = f"{ckpt_type}_epoch{epoch+1}_loss{loss:.4f}_mae{mae:.4f}.pth"
        save_path = os.path.join(self.model_path, 'weights', filename)
        torch.save(state, save_path)
        
        if self.save_opt:
            opt_state = {
                'optimizer_state': self.optimizer.state_dict(),
                **state
            }
            opt_path = os.path.join(self.model_path, 'optimizers', filename)
            torch.save(opt_state, opt_path)
            
        print(f"Checkpoint saved: {filename}")
            
    def _save_training_metrics(self):
        """保存训练指标"""
        # 保存完整训练损失
        np.savetxt(
            os.path.join(self.model_path, 'metrics', 'full_train_loss.csv'),
            np.array(self.train_losses),
            delimiter=',',
            header="train_loss",
            comments='',
            fmt='%.6f'
        )
        
        # 保存对齐指标
        if len(self.val_epochs) > 0:
            aligned_data = np.column_stack((
                np.array(self.val_epochs)+1,
                [self.train_losses[e] for e in self.val_epochs],
                self.test_losses,
                self.mae_scores,
                self.f1_scores
            ))
            np.savetxt(
                os.path.join(self.model_path, 'metrics', 'aligned_metrics.csv'),
                aligned_data,
                delimiter=',',
                header="epoch,train_loss,test_loss,mae,f1_score",
                comments='',
                fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f']
            )
        
    def _plot_metrics(self):
        """绘制训练指标曲线"""
        if len(self.val_epochs) == 0:
            print("No validation metrics to plot!")
            return
            
        epochs = np.array(self.val_epochs) + 1  # 转换为从1开始的epoch编号
        
        plt.figure(figsize=(15, 5))
        
        # ====== 子图1：损失曲线 ====== #
        plt.subplot(1, 3, 1)
        # 绘制完整训练曲线（浅灰色背景）
        plt.plot(range(1, len(self.train_losses)+1), self.train_losses, 
                color='lightgray', linestyle='-', label='Full Train Loss')
        # 高亮显示验证点的训练损失
        plt.plot(epochs, [self.train_losses[e] for e in self.val_epochs], 
                color='royalblue', marker='o', label='Train Loss at Val Points')
        # 验证损失
        plt.plot(epochs, self.test_losses, 
                color='darkorange', marker='s', label='Validation Loss')
        plt.title("Training Dynamics")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ====== 子图2：MAE曲线 ====== #
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.mae_scores, 
                color='forestgreen', marker='^', linestyle='-')
        plt.title("MAE Progression")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.grid(True, alpha=0.3)
        
        # ====== 子图3：F1曲线 ====== #
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.f1_scores, 
                color='crimson', marker='D', linestyle='-')
        plt.title("F1 Score Progression")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.model_path, 'metrics', 'training_metrics.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print("Training metrics visualization saved.")

if __name__ == '__main__':
    config = parse_arguments()
    trainer = TrainingEngine(config)
    trainer.train()