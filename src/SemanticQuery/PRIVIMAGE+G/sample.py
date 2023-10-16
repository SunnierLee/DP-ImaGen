import torch
import os
import numpy as np
from PIL import Image
import argparse

from big_resnet import Generator


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_file", type=str, default="/bigtemp/fzv6en/kecen0923/BigGAN/cifar10/e10/cifar10_e10_top5_e50_update5/")
args = parser.parse_args()

exp_file = args.exp_file
weight = "{}/netG.pth".format(exp_file)
z_dim = 80
num_classes = 10
sample_num = 50000
batch_size = 2048
out_f = "{}/sample{}".format(exp_file, sample_num)
G = Generator()
G.load_state_dict(torch.load(weight))
G = G.cuda()
G.eval()

os.mkdir(out_f)

for i in range(num_classes):
    os.mkdir(os.path.join(out_f, str(i).zfill(6)))

sample_round = sample_num // batch_size + 1
labels_count = [0 for _ in range(num_classes)]
images_count = 0
with torch.no_grad():
    for _ in range(sample_round):

        z = torch.randn((batch_size, z_dim)).cuda()
        labels = torch.randint(0, num_classes, size=(batch_size, )).cuda()
        x = G(z, labels)
        x = (x / 2. + .5).clip(0., 1.)
        x = x.detach().cpu().permute(0, 2, 3, 1) * 255.
        x = x.numpy().astype(np.uint8)

        for i in range(batch_size):
            if images_count > sample_num:
                break
            image = x[i]
            label = labels[i].item()

            Image.fromarray(image).save(os.path.join(
                        out_f, str(label).zfill(6), str(labels_count[label]).zfill(6) + '.png'))
            
            labels_count[label] += 1
            images_count += 1



