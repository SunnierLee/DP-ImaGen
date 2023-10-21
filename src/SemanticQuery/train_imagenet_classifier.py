import argparse
import os
import time
import logging

import torch
from torch import nn
from torchvision import models
import torch.optim as optim
from torchvision.datasets import ImageFolder, CIFAR10, MNIST, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from SpecificImagenet import SpecificClassImagenet


criterion = nn.CrossEntropyLoss()


def train(net, loader, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct/total


def test(net, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct/total


class MyClassifier(nn.Module):
    def __init__(self, model="resnet", num_classes=1000):
        super(MyClassifier, self).__init__()
        if model == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)


def main(args):
    world_size = torch.cuda.device_count()
    args.ddp = world_size > 1
    if args.ddp:
        dist.init_process_group("nccl", init_method='env://')
        rank = dist.get_rank()
        rank = rank % world_size
        torch.cuda.set_device(rank)
    else:
        rank = 0

    if rank == 0:
        t = time.localtime()
        main_name = 'train_imagenet_classifier'
        exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(main_name, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min,
                                                 t.tm_sec)
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)
            os.mkdir('{}/weights'.format(exp_name))

        gfile_stream = open(os.path.join(exp_name, 'log.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        logging.info(args)

    batch_size = args.batch_size // world_size
    val_batch_size = args.val_batch_size // world_size
    num_workers = args.num_workers // world_size

    model = MyClassifier(model=args.model).to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    data_dir = args.data_dir
    train_dataset = SpecificClassImagenet(root=data_dir, specific_class=None, transform=transforms.Compose([
                #CenterCropLongEdge(),
                #transforms.Resize(size=(config.data.resolution, config.data.resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]))
    val_dataset = SpecificClassImagenet(root=data_dir, split="val", specific_class=None, transform=transforms.Compose([
                #CenterCropLongEdge(),
                #transforms.Resize(size=(config.data.resolution, config.data.resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]))

    if rank == 0:
        print(len(train_dataset))
        print(len(val_dataset))

    if args.ddp:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=DistributedSampler(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                drop_last=False)

    best_acc = 0
    for epoch in range(args.epoch):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        train_acc = train(model, train_loader, optimizer, rank)
        if rank == 0:
            print('Epoch: {} Train Acc: {}'.format(epoch, train_acc))
            logging.info('Epoch: {} Train Acc: {}'.format(epoch, train_acc))

            val_acc = test(model, val_loader, rank)
            logging.info('Val Acc: {}'.format(val_acc))
            print('Val Acc: {}'.format(val_acc))
            if val_acc > best_acc:
                logging.info('Saving..')
                print('Saving..')
                torch.save(model.state_dict(), '{}/weights/{:.3f}_ckpt.pth'.format(exp_name, val_acc))
                best_acc = val_acc
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--data_dir', type=str, default='/data_dir/imagenet32/')
    parser.add_argument('--img_size', type=int, default=32)

    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--val_batch_size', type=int, default=8192*2)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ddp', type=bool, default=False)

    args = parser.parse_args()

    main(args)