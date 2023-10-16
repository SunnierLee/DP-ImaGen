#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs DCGAN training with differential privacy.

"""
from __future__ import print_function

import argparse
import os
import random
import logging

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import pickle

#from stylegan3.dataset import ImageFolderDataset
from utils_ import compute_fid, set_logger, Wrap_D, Wrap_D_DDP, Wrap_G, Wrap_G_DDP
from dnnlib.util import open_url
from big_resnet import Generator, Discriminator
from stylegan3.dataset import ImageFolderDataset


def main(opt):
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    gfile_stream = open(os.path.join(opt.outf, 'stdout.txt'), 'w')
    set_logger(gfile_stream)
    logging.info(opt)

    dataset = dset.CIFAR10(
    root=opt.data_root,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ))
    num_classes = 10

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        #num_workers=int(opt.workers),
        batch_size=opt.batch_size,
    )

    device = torch.device(opt.device)
    ngpu = int(opt.ngpu)

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(device)

    netG = Generator()

    model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Number of trainable parameters in G: {}".format(n_params))

    if opt.netG != "":
        netG.load_state_dict(torch.load(opt.netG))
    netG = netG.to(device)
    netG = Wrap_G(Wrap_G_DDP(netG), ngpu)

    #output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    netD = Discriminator()

    model_parameters = filter(lambda p: p.requires_grad, netD.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Number of trainable parameters in D: {}".format(n_params))

    if opt.netD != "":
        netD.load_state_dict(torch.load(opt.netD))
    netD = netD.to(device)
    netD = Wrap_D(Wrap_D_DDP(netD), ngpu)

    FIXED_NOISE = torch.randn(64, 80, device=device)
    FIXED_LABEL = torch.randint(low=0, high=num_classes, size=(64, ), device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))

    if not opt.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=opt.secure_rng)

        netD, optimizerD, dataloader = privacy_engine.make_private_with_epsilon(
            module=netD,
            optimizer=optimizerD,
            data_loader=dataloader,
            target_delta=opt.delta,
            target_epsilon=opt.epsilon,
            epochs=opt.epochs,
            max_grad_norm=opt.max_per_sample_grad_norm
        )

    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))


    def d_hinge(d_logit_real, d_logit_fake):
        return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


    def g_hinge(d_logit_fake):
        return -torch.mean(d_logit_fake)


    all_steps = 0
    for epoch in range(opt.epochs):
        with BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=opt.dp_max_physical_batch_size,
                optimizer=optimizerD,
                n_splits=opt.dp_n_splits if opt.dp_n_splits > 0 else None) as memory_safe_data_loader:
            for i, data in enumerate(memory_safe_data_loader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                optimizerD.zero_grad(set_to_none=True)

                real_images = data[0].to(device)
                real_labels = data[1].to(device)
                batch_size = real_images.size(0)

                fake_labels = torch.randint(0, num_classes, (batch_size, ), device=device)
                noise = torch.randn((batch_size, 80), device=device)
                fake_images = netG(noise, fake_labels)

                real_out = netD(real_images, real_labels)
                fake_out = netD(fake_images.detach(), fake_labels)

                # below, you actually have two backward passes happening under the hood
                # which opacus happens to treat as a recursive network
                # and therefore doesn't add extra noise for the fake samples
                # noise for fake samples would be unnecesary to preserve privacy

                errD = d_hinge(real_out, fake_out)
                errD.backward()
                optimizerD.step()
                optimizerD.zero_grad(set_to_none=True)

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if all_steps % (opt.d_updates * opt.dp_n_splits) == 0:
                    optimizerG.zero_grad()
                    batch_size = opt.batch_size // opt.dp_n_splits
                    for _ in range(opt.dp_n_splits):
                        optimizerD.zero_grad(set_to_none=True)
                        fake_labels = torch.randint(0, num_classes, (batch_size, ), device=device)
                        noise = torch.randn((batch_size, 80), device=device)
                        fake_images = netG(noise, fake_labels)
                        output_g = netD(fake_images, fake_labels)
                        errG = g_hinge(output_g) / opt.dp_n_splits
                        errG.backward()
                    optimizerG.step()
                
                all_steps += 1

                if not opt.disable_dp:
                    epsilon = privacy_engine.accountant.get_epsilon(delta=opt.delta)
                    print(
                        f"epoch: {epoch}, Loss_D: {errD.item()} "
                        f"Loss_G: {errG.item()} "
                        "(ε = %.2f, δ = %.2f)" % (epsilon, opt.delta)
                    )
                    if all_steps%opt.log_step == 0:
                        logging.info(f"epoch: {epoch}, Loss_D: {errD.item()} "
                            f"Loss_G: {errG.item()}"
                            "(ε = %.2f, δ = %.2f)" % (epsilon, opt.delta))
                else:
                    print(
                        f"epoch: {epoch}, Loss_D: {errD.item()} "
                        f"Loss_G: {errG.item()}"
                    )
                    if all_steps%opt.log_step == 0:
                        logging.info(f"epoch: {epoch}, Loss_D: {errD.item()} "
                            f"Loss_G: {errG.item()}")

                if all_steps%opt.log_step == 0:
                    vutils.save_image(
                        real_images[:opt.sample_num], "%s/real_samples.png" % opt.outf, normalize=True
                    )
                    fake_images = netG(FIXED_NOISE, FIXED_LABEL)
                    vutils.save_image(
                        fake_images.detach()[:opt.sample_num],
                        "%s/fake_samples_iter_%03d.png" % (opt.outf, all_steps),
                        normalize=True,
                    )

                if all_steps%opt.fid_step == 0:
                    netG.eval()
                    def generator():
                        num_sampling_rounds = opt.dp_n_splits
                        batch_size = opt.sample_fid // num_sampling_rounds + 1
                        for _ in range(num_sampling_rounds):
                            noise = torch.randn(batch_size, 80, device=device)
                            fake_label = torch.randint(low=0, high=num_classes, size=(batch_size, ), device=device)
                            x = netG(noise, fake_label)
                            x = (x / 2. + .5).clip(0., 1.)
                            x = (x * 255.).to(torch.uint8)
                            yield x
                    with torch.no_grad():
                        fid = compute_fid(generator(), inception_model, opt.stats_path, n_samples=opt.sample_fid, 
                                        device=device)
                        logging.info("Epoch: {}, FID: {}".format(epoch, fid))
                    netG.train()

                    # do checkpointing
                    torch.save(netG.wrap_model.model.state_dict(), "%s/netG.pth" % (opt.outf))
                    torch.save(netD.wrap_model.model.state_dict(), "%s/netD.pth" % (opt.outf))

    netG.eval()
    def generator():
        num_sampling_rounds = opt.dp_n_splits
        batch_size = opt.sample_fid // num_sampling_rounds + 1
        for _ in range(num_sampling_rounds):
            noise = torch.randn(batch_size, 80, device=device)
            fake_label = torch.randint(low=0, high=num_classes, size=(batch_size, ), device=device)
            x = netG(noise, fake_label)
            x = (x / 2. + .5).clip(0., 1.)
            x = (x * 255.).to(torch.uint8)
            yield x
    with torch.no_grad():
        fid = compute_fid(generator(), inception_model, opt.stats_path, n_samples=opt.sample_fid, 
                        device=device)
        logging.info("Epoch: {}, FID: {}".format(epoch, fid))
    netG.train()

    # do checkpointing
    torch.save(netG.wrap_model.model.state_dict(), "%s/netG.pth" % (opt.outf))
    torch.save(netD.wrap_model.model.state_dict(), "%s/netD.pth" % (opt.outf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="/bigtemp/fzv6en/datasets/")
    parser.add_argument("--stats_path", type=str, default="/bigtemp/fzv6en/datasets/cifar10.npz")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=2
    )
    parser.add_argument("--batch_size", type=int, default=16384, help="input batch size")
    parser.add_argument(
        "--imageSize",
        type=int,
        default=32,
        help="the height / width of the input image to network",
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train for"
    )
    parser.add_argument(
        "--d_updates", type=int, default=5
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0005, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    #parser.add_argument("--netG", default="", help="path to netG (to continue training)")
    #parser.add_argument("--netD", default="", help="path to netD (to continue training)")
    #parser.add_argument("--netG", default="/u/fzv6en/kecen/BigGAN/cifar10/pretrain/pretrain_cifar10_all_e1/netG.pth", help="path to netG (to continue training)")
    #parser.add_argument("--netD", default="/u/fzv6en/kecen/BigGAN/cifar10/pretrain/pretrain_cifar10_all_e1/netD.pth", help="path to netD (to continue training)")
    parser.add_argument("--netG", default="/u/fzv6en/kecen/BigGAN/pretrain_cifar10_top10/netG.pth", help="path to netG (to continue training)")
    parser.add_argument("--netD", default="/u/fzv6en/kecen/BigGAN/pretrain_cifar10_top10/netD.pth", help="path to netD (to continue training)")
    parser.add_argument(
        "--outf", default="cifar10_e10_top10_e50_update5", help="folder to output images and model checkpoints"
    )
    parser.add_argument("--manualSeed", type=int, help="manual seed", default=1)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )

    parser.add_argument(
        "--disable_dp",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure_rng",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=9.99,
        metavar="E",
        help="Privacy budget",
    )
    parser.add_argument(
        "-c",
        "--max_per_sample_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=8e-6,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--dp_max_physical_batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--dp_n_splits",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--sample_num",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--sample_fid",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--fid_step",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    main(args)