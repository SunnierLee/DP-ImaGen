import torch
import os
import torchvision
import argparse

from big_resnet import Generator


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_file", type=str, default="cifar10_e10_all_e50_update5")
parser.add_argument("--img_size", type=int, default=32)
parser.add_argument("--g_conv_dim", type=int, default=96)
args = parser.parse_args()

exp_file = args.exp_file
weight = "{}/netG.pth".format(exp_file)
z_dim = 80
num_classes = 10
sample_num = 8
batch_size = num_classes * sample_num
out_f = "{}/sample_show".format(exp_file)
G = Generator(img_size=args.img_size, g_conv_dim=args.g_conv_dim)
G.load_state_dict(torch.load(weight))
G = G.cuda()
G.eval()

os.mkdir(out_f)


with torch.no_grad():
    for j in range(64):
        z = torch.randn((batch_size, z_dim)).cuda()
        labels =  torch.tensor([[i] * sample_num for i in range(num_classes)]).cuda().view(-1)
        x = G(z, labels)
        x = (x / 2. + .5).clip(0., 1.)
        torchvision.utils.save_image(x.detach().cpu(), os.path.join(exp_file, "sample_show", 'batch_{}.png'.format(j)), padding=0, nrow=8)



