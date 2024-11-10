import argparse

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.datasets import ImageFolder, CIFAR10, MNIST, CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from classifier_models import resnet


def load_weight(net, weight_path):
    weight = torch.load(weight_path)
    weight = {k.replace('module.', ''): v for k, v in weight.items()}
    net.load_state_dict(weight)


class MyClassifier(nn.Module):
    def __init__(self, model="resnet", num_classes=1000):
        super(MyClassifier, self).__init__()
        if model == "resnet50":
            self.model = resnet.ResNet50(num_classes=num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def main(args):
    model = MyClassifier(model=args.model)

    load_weight(model, args.weight_file)
    model = model.cuda()
    model.eval()

    if args.tar_dataset == "cifar10":
        dataset = CIFAR10(root=args.data_dir, download=True, transform=transforms.Compose(
            [transforms.Resize(args.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)
             ]
        ))
    elif args.tar_dataset == "celeba":
        dataset = CelebA(root=args.data_dir, download=True, transform=transforms.Compose(
            [CenterCropLongEdge(),
             transforms.Resize(args.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)
             ]
        ))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
    semantics_hist = torch.zeros((args.tar_num_classes, args.ref_num_classes)).cuda()

    for (x, y) in loader:
        x = x.cuda()
        y = y.cuda()
        out = model(x)
        words_idx = torch.topk(out, k=args.num_words, dim=1)[1]
        for i in range(x.shape[0]):
            cls = y[i] if args.tar_dataset != "celeba" else 0
            words = words_idx[i]
            semantics_hist[cls, words] += 1

    sensitivity = np.sqrt(args.num_words)
    #sigma = np.sqrt(2 * np.log(1.25/args.delta)) / args.epsilon
    sigma = args.sigma1

    semantics_hist = semantics_hist + torch.randn_like(semantics_hist) * sensitivity * sigma

    cls_dict = {}
    for i in range(config.sensitive_data.n_classes):
        semantics_hist_i = semantics_hist[i]
        if i != 0:
            semantics_hist_i[topk_mask] = -999
        semantics_description_i = torch.topk(semantics_hist_i, k=config.public_data.selective.num_words)[1]
        if i == 0:
            topk_mask = semantics_description_i
        else:
            topk_mask = torch.cat([topk_mask, semantics_description_i])
        cls_dict[i] = list(semantics_description_i.detach().cpu().numpy())

    torch.save(cls_dict, "QueryResults/chosen_top{}_classes_for_{}_epsilon{}.pth".format(args.num_words, args.tar_dataset, args.epsilon))
    print(cls_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default='')
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--num_words', type=int, default=50)
    #parser.add_argument('--epsilon', type=float, default=0.01)
    #parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--sigma1', type=float, default=484)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ref_num_classes', type=int, default=1000)
    parser.add_argument('--tar_num_classes', type=int, default=10)
    parser.add_argument('--tar_dataset', type=str, default='cifar10')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4096*2)

    args = parser.parse_args()
    main(args)
