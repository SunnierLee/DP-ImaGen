import os
import logging
import torch
import numpy as np
import torch.distributed as dist
import pickle
import torchvision
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.ncsnpp import NCSNpp
from stylegan3.dataset import ImageFolderDataset
from SpecificImagenet import SpecificClassImagenet
from utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid
from dnnlib.util import open_url
from model.ema import ExponentialMovingAverage
from score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from samplers import ddim_sampler, edm_sampler


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return torchvision.transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def training(config, workdir, mode):
    set_seeds(config.setup.global_rank, config.train.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    fid_dir = os.path.join(workdir, 'fid')

    if config.setup.global_rank == 0:
        if mode == 'train':
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
            make_dir(fid_dir)
    dist.barrier()

    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vpsde':
        if config.model.denoiser_network == 'song':
            model = VPSDEDenoiser(config.model.beta_min, config.model.beta_max - config.model.beta_min,
                                  config.model.scale, NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vesde':
        if config.model.denoiser_network == 'song':
            model = VESDEDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'v':
        if config.model.denoiser_network == 'song':
            model = VDenoiser(
                NCSNpp(**config.model.network).to(config.setup.device))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model = DDP(model)
    if config.model.ckpt is not None:
        state = torch.load(config.model.ckpt, map_location=config.setup.device)
        logging.info(model.load_state_dict(state['model'], strict=True))
        
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)

    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
    else:
        raise NotImplementedError

    state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

    if config.setup.global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
        logging.info('Number of total epochs: %d' % config.train.n_epochs)
        logging.info('Starting training at step %d' % state['step'])
    dist.barrier()

    if config.data.name.startswith('fmnist'):
        dataset = torchvision.datasets.FashionMNIST(
            root='toy_data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    elif config.data.name.startswith('mnist'):
        dataset = torchvision.datasets.MNIST(
            root='toy_data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    elif config.data.name.startswith('pretrain'):
        if config.data.specific_class is not None:
            specific_class = torch.load(config.data.specific_class)
        else:
            specific_class = None
        dataset = SpecificClassImagenet(root=config.data.path, specific_class=specific_class, transform=transforms.Compose([
                #CenterCropLongEdge(),
                #transforms.Resize(size=(config.data.resolution, config.data.resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]))
        if config.setup.global_rank == 0:
            logging.info(
                'Number of images: {}'.format(len(dataset)))
        # dataset = torchvision.datasets.ImageFolder(
        #     root=config.data.path, transform=transforms.Compose([
        #         CenterCropLongEdge(),
        #         transforms.Resize(size=(config.data.resolution, config.data.resolution)),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor(),
        #     ]))
    # elif config.data.name.startswith('celeba'):
    #     dataset = torchvision.datasets.CelebA(
    #         root=config.data.path, split="all", download=True, transform=torchvision.transforms.Compose(
    #         [CenterCropLongEdge(),
    #          torchvision.transforms.Resize([config.data.resolution, config.data.resolution]),
    #          torchvision.transforms.ToTensor(),
    #          torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)
    #          ]
    #         )
    #     )
    else:
        dataset = ImageFolderDataset(
            config.data.path, config.data.resolution, **config.data.dataset_params)

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config.train.batch_size, sampler=DistributedSampler(dataset), num_workers=config.data.dataloader_params.num_workers,
        pin_memory=config.data.dataloader_params.pin_memory)

    if config.loss.version == 'edm':
        loss_fn = EDMLoss(**config.loss).get_loss
    elif config.loss.version == 'vpsde':
        loss_fn = VPSDELoss(**config.loss).get_loss
    elif config.loss.version == 'vesde':
        loss_fn = VESDELoss(**config.loss).get_loss
    elif config.loss.version == 'v':
        loss_fn = VLoss(**config.loss).get_loss
    else:
        raise NotImplementedError

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(config.setup.device)

    def sampler(x, y=None):
        if config.sampler.type == 'ddim':
            return ddim_sampler(x, y, model, **config.sampler)
        elif config.sampler.type == 'edm':
            return edm_sampler(x, y, model, **config.sampler)
        else:
            raise NotImplementedError

    snapshot_sampling_shape = (config.sampler.snapshot_batch_size,
                               config.data.num_channels, config.data.resolution, config.data.resolution)
    fid_sampling_shape = (config.sampler.fid_batch_size, config.data.num_channels,
                          config.data.resolution, config.data.resolution)

    for epoch in range(config.train.n_epochs):
        dataset_loader.sampler.set_epoch(epoch)

        for _, (train_x, train_y) in enumerate(dataset_loader):
            if state['step'] % config.train.snapshot_freq == 0 and state['step'] >= config.train.snapshot_threshold and config.setup.global_rank == 0:
                logging.info(
                    'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                model.eval()
                with torch.no_grad():
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                        sample_dir, 'iter_%d' % state['step']), config.setup.device, config.data.n_classes)
                    ema.restore(model.parameters())
                model.train()

                save_checkpoint(os.path.join(
                    checkpoint_dir, 'snapshot_checkpoint.pth'), state)
            dist.barrier()

            if state['step'] % config.train.fid_freq == 0 and state['step'] >= config.train.fid_threshold:
                model.eval()
                with torch.no_grad():
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    fids = compute_fid(config.train.fid_samples, config.setup.global_size, fid_sampling_shape, sampler, inception_model,
                                       config.data.fid_stats, config.setup.device, config.data.n_classes)
                    ema.restore(model.parameters())

                    if config.setup.global_rank == 0:
                        for i, fid in enumerate(fids):
                            logging.info('FID %d at iteration %d: %.6f' % (
                                i, state['step'], fid))
                    dist.barrier()
                model.train()

            if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_threshold and config.setup.global_rank == 0:
                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                save_checkpoint(checkpoint_file, state)
                logging.info(
                    'Saving  checkpoint at iteration %d' % state['step'])
            dist.barrier()

            if config.data.name.startswith('mnist') or config.data.name.startswith('fmnist') or config.data.name.startswith('pretrain'):
                x = (train_x.to(config.setup.device).to(
                    torch.float32) * 2. - 1.)
            else:
                x = (train_x.to(config.setup.device).to(
                    torch.float32) / 127.5 - 1.)

            if config.data.n_classes is None:
                y = None
            else:
                if config.data.one_hot:
                    train_y = torch.argmax(train_y, dim=1)
                y = train_y.to(config.setup.device)
                if y.dtype == torch.float32:
                    y = y.long()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(loss_fn(model, x, y))
            loss.backward()
            optimizer.step()

            if (state['step'] + 1) % config.train.log_freq == 0 and config.setup.global_rank == 0:
                logging.info('Loss: %.4f, step: %d' %
                             (loss, state['step'] + 1))
            dist.barrier()

            state['step'] += 1
            state['ema'].update(model.parameters())

        logging.info('After %d epochs' % (epoch + 1))

    if config.setup.global_rank == 0:
        checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
        save_checkpoint(checkpoint_file, state)
        logging.info('Saving final checkpoint.')
    dist.barrier()

    model.eval()
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        fids = compute_fid(config.train.fid_samples, config.setup.global_size, fid_sampling_shape, sampler, inception_model,
                           config.data.fid_stats, config.setup.device, config.data.n_classes)
        ema.restore(model.parameters())

    if config.setup.global_rank == 0:
        for i, fid in enumerate(fids):
            logging.info('Final FID %d: %.6f' % (i + 1, fid))
    dist.barrier()
