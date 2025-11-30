import os
import sys
import numpy as np
from torchvision import datasets, disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2
from torch.utils.data import default_collate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from resnet import ResNet
import argparse
import wandb
import torchinfo
import onnx
from onnxsim import simplify

mixup = v2.MixUp(num_classes=10)

def mixup_collate_fn(batch):
        return mixup(*default_collate(batch))


def conf_collate_fn(mixup):
    if mixup:
        return mixup_collate_fn
    else:
        return default_collate

def main(args):
    run = wandb.init(project="explore", config=args)
    lr = wandb.config['learning_rate']
    epochs = wandb.config['epochs']
    batch_size = wandb.config['batch_size']
    momentum = wandb.config['momentum']
    weight_decay = wandb.config['weight_decay']
    mixup = wandb.config['mixup']
    depth = wandb.config['depth']
    width = wandb.config['width']
    groups = wandb.config['groups']
    chkpt_path = wandb.config['chkpt_path']

    preproc = v2.Compose([
        v2.PILToTensor(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True)
    ])

    collate_fn = conf_collate_fn(args)

    # download CIFAR10 dataset
    data_path = os.path.join(os.path.dirname(__file__), './datasets/')
    train_data = datasets.CIFAR10(data_path, train=True, download=True, transform=preproc)
    test_data = datasets.CIFAR10(data_path, train=False, download=True, transform=preproc)


    if depth == 18:
        model = ResNet([(width, 1, [groups, groups]), (width, 1, [groups, groups]), (width*2, 2, [groups, groups]), (width*2, 1, [groups, groups]), (width*4, 2, [groups, groups]), (width*4, 1, [groups, groups]), (width*8, 2, [groups, groups]), (width*8, 1, [groups, groups])])
    elif depth == 14:
        model = ResNet([(width, 1, [groups, groups]), (width, 1, [groups, groups]), (width*2, 2, [groups, groups]), (width*2, 1, [groups, groups]), (width*4, 2, [groups, groups]), (width*4, 1, [groups, groups])])
    elif depth == 8:
        model = ResNet([(width, 1, [groups, groups]), (width*2, 2, [groups, groups]), (width*4, 2, [groups, groups])])
    else:
        raise ValueError('Invalid depth')

    # export as onnx
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, "trained-model.onnx")

    # Load, simplify, and save
    model_onnx = onnx.load("trained-model.onnx")
    model_onnx, check = simplify(model_onnx)
    onnx.save(model_onnx, "trained-model.onnx")

    epochs = 150

    summary = torchinfo.summary(model, input_size=(32, 3, 32, 32))
    run.config['total_params'] = summary.total_params
    run.config['mult_add'] = summary.total_mult_adds

    collate_fn = conf_collate_fn(mixup)

    # train ResNet-18 model
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=conf_collate_fn(mixup))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()


    max_accuracy = 0
    for epoch in range(epochs):
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        max_accuracy = max_accuracy if max_accuracy > correct / total else correct / total
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': correct / total,
            'max_accuracy': max_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })
        print(f'Epoch: {epoch}, Test Accuracy: {correct / total}')
        model.train()

    torch.save(model.state_dict(), chkpt_path + run.id + '.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=int, default=0.9, help="Momentum")
    parser.add_argument("-wd", "--weight_decay", type=int, default=5e-4, help="Weight decay")
    parser.add_argument("--mixup", action="store_true", help="Use MixUp data augmentation")
    parser.add_argument("--depth", type=int, default=18, help="ResNet depth")
    parser.add_argument("--width", type=int, default=64, help="ResNet width")
    parser.add_argument("--groups", type=int, default=1, help="ResNet groups")
    parser.add_argument("--chkpt_path", type=str, default="./", help="Path to save model checkpoints")

    args = parser.parse_args()
    main(args)