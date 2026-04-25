import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as random
import torchvision
import kornia
import torchvision.transforms as transforms
from model.data import enhance_buffer_data, GaussianBlur

class replay_buffer:
    def __init__(self, max_size, replace_prob, img_shape):
        self.max_size = max_size
        self.img_shape = img_shape
        self.replace_prob = replace_prob
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        if len(self.buffer) < num_samples:
            self.buffer = torch.rand(num_samples, *self.img_shape)
        
        random_indexes = torch.randint(0, len(self.buffer), (num_samples,))

        random_indexes_mask = torch.rand(num_samples) < self.replace_prob

        self.buffer[random_indexes[random_indexes_mask]] = torch.rand_like(self.buffer[random_indexes[random_indexes_mask]])

        return self.buffer[random_indexes]

    def update_buffer(self, samples):

        self.buffer = torch.cat([samples, self.buffer])
        self.buffer = self.buffer[:self.max_size]

class ReplayBuffer:
    def __init__(self, max_size, replace_prob, img_shape, device=None, dtype=None):
        self.max_size = int(max_size)
        self.img_shape = tuple(img_shape)
        self.replace_prob = float(replace_prob)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.transform = enhance_buffer_data(img_shape[1])
        # 预分配存储 + 指针与当前有效大小
        self.buffer = torch.empty((self.max_size, *self.img_shape), device=self.device, dtype=self.dtype)
        self.ptr = 0          # 下次写入位置
        self.size = 0         # 已写入的有效样本数 (<= max_size)

    def __len__(self):
        return self.size

    def sample(self, num_samples: int, train=False):
        num_samples = int(num_samples)

        # 若当前有效数据不足以采样，先用随机噪声填充到至少 num_samples
        if self.size < num_samples:
            need = min(self.max_size, num_samples - self.size)
            # 写入随机数据
            end = (self.ptr + need) % self.max_size
            if end > self.ptr:
                self.buffer[self.ptr:end].uniform_(0, 1)
            else:
                first = self.max_size - self.ptr
                self.buffer[self.ptr:].uniform_(0, 1)
                self.buffer[:end].uniform_(0, 1)
            self.ptr = end
            self.size = min(self.max_size, self.size + need)

        # 现在可以从 [0, self.size) 均匀采样
        random_indexes = torch.randint(self.size, (num_samples,), device=self.device)

        # 按 replace_prob 替换这批被采样到的 buffer 位置（就地随机化）
        chosen_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        if self.replace_prob > 0:
            mask = torch.rand(num_samples, device=self.device) < self.replace_prob
            if mask.any():
                chosen = random_indexes[mask]
                chosen_mask[mask] = True
                # 直接对选中的位置写入新噪声，避免构造临时大张量
                self.buffer[chosen].uniform_(0, 1)
        
        # 获取样本
        samples = self.buffer[random_indexes].clone()
        
        # 只对不属于chosen的样本（即原始buffer样本）进行transform
        if self.transform is not None and train:
            # 找到需要transform的样本索引（不属于chosen的样本）
            transform_mask = ~chosen_mask
            
            transform_mask = [idx for idx,mask in enumerate(transform_mask) if mask]
            for mask in transform_mask:
                samples[mask] = self.transform(samples[mask])
            #if transform_mask.any():
                # 只对原始buffer样本进行transform
                #samples[transform_mask] = self.transform(samples[transform_mask])
        
        return samples

    def update_buffer(self, samples: torch.Tensor):
        """
        将 samples 写入缓冲区（覆盖最旧数据），O(写入量) 拷贝，无需 cat。
        要求 samples 形状为 (N, *img_shape)，设备与 dtype 可与 buffer 不同，会自动搬运。
        """
        if samples.ndim == len(self.img_shape):
            # 允许传入单个样本
            samples = samples.unsqueeze(0)
        assert samples.shape[1:] == self.img_shape, "expected shape (*,{}), got {}".format(self.img_shape, samples.shape)
        samples = samples.to(device=self.device, dtype=self.dtype, copy=False)

        n = samples.shape[0]
        if n >= self.max_size:
            # 只保留最后 max_size 个
            self.buffer.copy_(samples[-self.max_size:])
            self.ptr = 0
            self.size = self.max_size
            return

        end = (self.ptr + n) % self.max_size
        if end > self.ptr:
            self.buffer[self.ptr:end].copy_(samples)
        else:
            first = self.max_size - self.ptr
            self.buffer[self.ptr:].copy_(samples[:first])
            self.buffer[:end].copy_(samples[first:])
        self.ptr = end
        self.size = min(self.max_size, self.size + n)
