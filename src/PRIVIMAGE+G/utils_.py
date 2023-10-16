import torch
from torch import nn
from torch import autograd
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import PIL
from torchvision.utils import make_grid
from scipy import linalg
from pathlib import Path
import logging


class Wrap_G_DDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_y):
        x, y = x_y[:, :-1, ...], x_y[:, -1, ...].view(-1).long()
        return self.model(x, y)


class Wrap_G(nn.Module):
    def __init__(self, wrap_model, ngpu) -> None:
        super().__init__()
        self.wrap_model = wrap_model
        self.ngpu = ngpu
    
    def forward(self, x, y):
        y = y.view(-1, 1)
        x_y = torch.cat([x, y], dim=1)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.wrap_model, x_y, range(self.ngpu))
        else:
            output = self.wrap_model(x_y)
        return output


class Wrap_D_DDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_y):
        x, y = x_y[:, :-1, ...], x_y[:, -1, 0, 0].view(-1).long()
        return self.model(x, y)


class Wrap_D(nn.Module):
    def __init__(self, wrap_model, ngpu) -> None:
        super().__init__()
        self.wrap_model = wrap_model
        self.ngpu = ngpu
    
    def forward(self, x, y):
        y = y.view(-1, 1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x_y = torch.cat([x, y], dim=1)
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.wrap_model, x_y, range(self.ngpu))
        else:
            output = self.wrap_model(x_y)
        return output


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads


def cal_grad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_out = discriminator(interpolates, real_labels)
    grads = cal_deriv(inputs=interpolates, outputs=fake_out, device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def get_activations(dl, model, device, max_samples):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    for batch in dl:
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]

        batch = batch.to(device)
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            pred = model(batch.to(device),
                         return_features=True).unsqueeze(-1).unsqueeze(-1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
        total_processed += pred.shape[0]
        if max_samples is not None and total_processed > max_samples:
            print('Max of %d samples reached.' % max_samples)
            break

    pred_arr = np.concatenate(pred_arr, axis=0)
    if max_samples is not None:
        pred_arr = pred_arr[:max_samples]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fd


def compute_fid(samples, inception_model, stats_path, device, n_samples=None):
    act = get_activations(samples, inception_model, device=device, max_samples=n_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()
    #average_tensor(m)
    #average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    stats = np.load(stats_path)
    data_pools_mean = stats['mu']
    data_pools_sigma = stats['sigma']
    fid = calculate_frechet_distance(data_pools_mean,
                data_pools_sigma, all_pool_mean, all_pool_sigma)
    return fid