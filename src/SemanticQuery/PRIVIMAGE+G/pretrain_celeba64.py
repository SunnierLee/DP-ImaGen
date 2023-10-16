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
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
import pickle

#from stylegan3.dataset import ImageFolderDataset
from utils_ import compute_fid, set_logger
from dnnlib.util import open_url
from big_resnet import Generator, Discriminator
from stylegan3.dataset import ImageFolderDataset
from SpecificImagenet import SpecificClassImagenet


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

    if opt.specific_class != "":
        specific_class = torch.load(opt.specific_class)
    else:
        specific_class = None
    dataset = SpecificClassImagenet(root=opt.data_root, specific_class=specific_class, transform=transforms.Compose([
                #CenterCropLongEdge(),
                #transforms.Resize(size=(config.data.resolution, config.data.resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]))
    logging.info("number of images: {}".format(len(dataset)))
    num_classes = 1

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

    netG = Generator(img_size=opt.imageSize, g_conv_dim=64)
    netG = netG.to(device)
    netG = Wrap_G(Wrap_G_DDP(netG), ngpu)
    if opt.netG != "":
        netG.load_state_dict(torch.load(opt.netG))

    #output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    netD = Discriminator(img_size=opt.imageSize, d_conv_dim=64)
    netD = netD.to(device)
    netD = Wrap_D(Wrap_D_DDP(netD), ngpu)
    if opt.netD != "":
        netD.load_state_dict(torch.load(opt.netD))

    FIXED_NOISE = torch.randn(64, 80, device=device)
    FIXED_LABEL = torch.randint(low=0, high=num_classes, size=(64, ), device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))


    def d_hinge(d_logit_real, d_logit_fake):
        return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


    def g_hinge(d_logit_fake):
        return -torch.mean(d_logit_fake)


    all_steps = 0
    for epoch in range(opt.epochs):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            optimizerD.zero_grad(set_to_none=True)

            real_images = data[0].to(device)
            #real_labels = data[1].to(device)
            batch_size = real_images.size(0)
            real_labels = torch.zeros(batch_size, device=device).long()

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
            if all_steps % opt.d_updates == 0:
                optimizerG.zero_grad()
                output_g = netD(fake_images, fake_labels)
                errG = g_hinge(output_g)
                errG.backward()
                optimizerG.step()
            
            all_steps += 1

            print(
                f"epoch: {epoch}, Loss_D: {errD.item()} "
                f"Loss_G: {errG.item()}"
            )
            if all_steps%opt.log_step == 0:
                logging.info(f"epoch: {epoch}, Loss_D: {errD.item()} "
                    f"Loss_G: {errG.item()}")

            if all_steps%opt.fid_step == 0:
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
                    num_sampling_rounds = opt.sample_fid // opt.batch_size + 1
                    for _ in range(num_sampling_rounds):
                        noise = torch.randn(opt.batch_size, 80, device=device)
                        fake_label = torch.randint(low=0, high=num_classes, size=(opt.batch_size, ), device=device)
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
        num_sampling_rounds = opt.sample_fid // opt.batch_size + 1
        for _ in range(num_sampling_rounds):
            noise = torch.randn(opt.batch_size, 80, device=device)
            fake_label = torch.randint(low=0, high=num_classes, size=(opt.batch_size, ), device=device)
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
    parser.add_argument("--dataset", type=str, default="celeba64")
    parser.add_argument("--data_root", type=str, default="/bigtemp/fzv6en/datasets/imagenet64/")
    parser.add_argument("--stats_path", type=str, default="/bigtemp/fzv6en/datasets/celeba_64.npz")
    parser.add_argument(
    "--specific_class",
    type=str,
    default="/u/fzv6en/kecen/RFF_DDPM/chosen_top500_classes_for_celeba_epsilon0.001.pth"
)
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=2
    )
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--imageSize",
        type=int,
        default=64,
        help="the height / width of the input image to network",
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train for"
    )
    parser.add_argument(
        "--d_updates", type=int, default=5
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--netG", default="", help="path to netG (to continue training)")
    parser.add_argument("--netD", default="", help="path to netD (to continue training)")
    parser.add_argument(
        "--outf", default="pretrain_celeba64_top500", help="folder to output images and model checkpoints"
    )
    parser.add_argument("--manualSeed", type=int, help="manual seed", default=0)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
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
        default=5000,
    )

    args = parser.parse_args()

    main(args)