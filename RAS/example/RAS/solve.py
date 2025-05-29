from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print('input + output channels need to be the same')
            raise
        if h != w:
            print('filters need to be square')
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# === 初始化 ===
base_weights = '5stage-vgg.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# 设置上采样层为双线性插值初始化
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# 加载预训练权重
solver.net.copy_from(base_weights)

# === 训练参数 ===
num_epochs = 20
num_train_images = 700
batch_size = 10
iters_per_epoch = 70

# === 记录每个 side loss ===
loss_names = ['loss1-score', 'loss2-score', 'loss3-score',
              'loss4-score', 'loss5-score', 'loss6-score']
loss_histories = {name: [] for name in loss_names}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    epoch_loss = {name: 0.0 for name in loss_names}

    for i in range(iters_per_epoch):
        solver.step(1)
        for name in loss_names:
            epoch_loss[name] += solver.net.blobs[name].data

    for name in loss_names:
        avg_loss = epoch_loss[name] / iters_per_epoch
        loss_histories[name].append(avg_loss)
        print(f"{name}: {avg_loss:.4f}")

# === 画图 ===
plt.figure(figsize=(10, 6))
for name in loss_names:
    plt.plot(range(1, num_epochs + 1), loss_histories[name], label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Side Losses per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_per_side_output.png')
plt.show()
