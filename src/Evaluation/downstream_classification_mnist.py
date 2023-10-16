import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR10, ImageFolder, MNIST
import torchvision.transforms as transforms

import argparse
import logging
import os

from sklearn import linear_model, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from resnet9 import ResNet9


def cnn_classify(train_set, test_set, fp, num_classes=10):
    batch_size = 512
    lr = 0.1
    max_epoch = 50
    criterion = nn.CrossEntropyLoss()
    model = ResNet9(num_classes, 1)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, drop_last=False)

    model.train()
    for epoch in range(max_epoch): 
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), os.path.join(fp, "trained_cnn_weight.pth"))
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc


def mlp_classify(train_set, test_set):
    model = neural_network.MLPClassifier(max_iter=1000)
    
    train_num = len(train_set)
    test_num = len(test_set)
    train_loader = DataLoader(train_set, batch_size=train_num)
    test_loader = DataLoader(test_set, batch_size=test_num)
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.numpy(), y_train.numpy()
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.numpy(), y_test.numpy()
    x_train = x_train.reshape(train_num, -1)
    x_test = x_test.reshape(test_num, -1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    return acc


def logreg_classify(train_set, test_set):
    model = linear_model.LogisticRegression(solver='lbfgs', max_iter=50000, multi_class='auto')
    
    train_num = len(train_set)
    test_num = len(test_set)
    train_loader = DataLoader(train_set, batch_size=train_num)
    test_loader = DataLoader(test_set, batch_size=test_num)
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.numpy(), y_train.numpy()
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.numpy(), y_test.numpy()
    x_train = x_train.reshape(train_num, -1)
    x_test = x_test.reshape(test_num, -1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    return acc


def main(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("{}/evaluation_downstream_acc_log.txt".format(args.out_dir), mode='a')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(args)

    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_set = ImageFolder(root=args.train_dir, transform=transform_train)
    print(len(train_set))
    if args.dataset == "cifar10":
        test_set = MNIST(root=args.test_dir, train=False, transform=transform_test)
        num_classes = 10
    else:
        raise NotImplementedError

    if args.model == "cnn":
        acc = cnn_classify(train_set, test_set, args.out_dir, num_classes)
    elif args.model == "logreg":
        acc = logreg_classify(train_set, test_set)
    elif args.model == "mlp":
        acc = mlp_classify(train_set, test_set)
    else:
        raise NotImplementedError

    logger.info("Final acc: {}".format(acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/bigtemp/fzv6en/kecen0923/DPDM/mnist/mnist_28_e10_384/")
    parser.add_argument("--train_dir", type=str, default="/bigtemp/fzv6en/kecen0923/DPDM/mnist/mnist_28_e10_384/sample50000/samples/")
    parser.add_argument("--test_dir", type=str, default="/bigtemp/fzv6en/datasets/")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "mlp", "cnn"])

    args = parser.parse_args()
    model_list = ["cnn", "logreg", "mlp"]
    for model_ in model_list:
        args.model = model_
        main(args)