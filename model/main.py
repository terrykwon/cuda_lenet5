import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * \
            self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def test_accuracy(net, testloader):
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(testloader, desc='accuracy', position=1, leave=False)):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train(data_dir='data_dir', save_dir='save', learning_rate=1e-3, weight_decay=1e-4, label_smoothing=0.1, epoch=100):
    print("[INFO] Get CIFAR-10 dataloader")
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True)
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    print("[INFO] Prepare net, optim, lr_scheduler, loss")
    net = LeNet5().cuda()
    criterion = LabelSmoothLoss(smoothing=label_smoothing).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    print("[INFO] Start training")
    writer = SummaryWriter()
    net.train()
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    for epoch in tqdm(range(1, epoch + 1), desc='Train Epoch', position=0, leave=True):
        loss_average = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader, desc='training', position=1, leave=False)):
            # Learn
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Calculate Loss and train accuracy
            with torch.no_grad():
                loss_average += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Write some values
        writer.add_scalar('loss', loss_average / i, global_step=epoch)
        writer.add_scalar('train_accuracy', 100 * correct / total, global_step=epoch)
        writer.add_scalar('test_accuracy', test_accuracy(net, testloader), global_step=epoch)
        writer.flush()
        # Save checkpoint
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f"{save_dir}/lenet5-cifar10-epoch_{epoch}.pth")
        # Step scheduler
        scheduler.step()


def test(data_dir='data_dir', checkpoint_path='save/lenet5-cifar10-epoch_50.pth', index=0, batch=1):
    print(f"[INFO] Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = LeNet5().cuda()
    net.load_state_dict(checkpoint['net'])
    print(f"[INFO] Set activation hook")
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.conv1.register_forward_hook(get_activation('conv1'))
    net.conv2.register_forward_hook(get_activation('conv2'))
    net.fc1.register_forward_hook(get_activation('fc1'))
    net.fc2.register_forward_hook(get_activation('fc2'))
    net.fc3.register_forward_hook(get_activation('fc3'))
    print(f"[INFO] Get CIFAR-10 dataloader")
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False, num_workers=2)
    print(f"[INFO] Run with index {index}")
    for i, data in enumerate(testloader):
        if i * batch <= index and index < (i + 1) * batch:
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
            break
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"[INFO] Predict as {classes[predicted.item()]}({predicted.item()})")
    print(f"[INFO] Save conv1 with shape: {activation['conv1'].shape}")
    np.savetxt(f"save.v4/conv1_index_{index}.txt", activation['conv1'].cpu().view(6, -1))
    print(f"[INFO] Save conv2 with shape: {activation['conv2'].shape}")
    np.savetxt(f"save.v4/conv2_index_{index}.txt", activation['conv2'].cpu().view(16, -1))
    print(f"[INFO] Save fc1 with shape: {activation['fc1'].shape}")
    np.savetxt(f"save.v4/fc1_index_{index}.txt", activation['fc1'].cpu().view(120, -1))
    print(f"[INFO] Save fc2 with shape: {activation['fc2'].shape}")
    np.savetxt(f"save.v4/fc2_index_{index}.txt", activation['fc2'].cpu().view(84, -1))
    print(f"[INFO] Save fc3 with shape: {activation['fc3'].shape}")
    np.savetxt(f"save.v4/fc2_index_{index}.txt", activation['fc2'].cpu().view(84, -1))
    np.savetxt(f"save.v4/fc3_index_{index}.txt", activation['fc3'].cpu().view(10, -1))
    np.savetxt(f"save.v4/fc3_index_{index}.txt", activation['fc3'].cpu().view(10, -1))


def convert(checkpoint_path='save/lenet5-cifar10-epoch_50.pth', output_path='save/values.txt'):
    print(f"[INFO] Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = LeNet5()
    net.load_state_dict(checkpoint['net'])
    print(f"[INFO] Write weights and biases to {output_path}")
    with open(output_path, 'w') as f:
        # Save weights of conv
        f.write(f"conv1.weight: {net.conv1.weight.shape}\n\n")
        for oc in range(6):
            for ic in range(3):
                for i in range(5):
                    for j in range(5):
                        f.write(f"{net.conv1.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")
        f.write(f"conv2.weight: {net.conv2.weight.shape}\n\n")
        for oc in range(16):
            for ic in range(6):
                for i in range(5):
                    for j in range(5):
                        f.write(f"{net.conv2.weight[oc][ic][i][j]} ")
                    f.write("\n")
                f.write("\n")
        # Save biases of conv
        f.write(f"conv1.bias: {net.conv1.bias.shape}\n\n")
        for bias in net.conv1.bias:
            f.write(f"{bias} ")
        f.write("\n\n")
        f.write(f"conv2.bias: {net.conv2.bias.shape}\n\n")
        for bias in net.conv2.bias:
            f.write(f"{bias} ")
        f.write("\n\n")
        # Save weights of fc
        f.write(f"fc1.weight: {net.fc1.weight.shape}\n\n")
        for i in range(120):
            for j in range(400):
                f.write(f"{net.fc1.weight[i][j]} ")
            f.write("\n")
        f.write("\n")
        f.write(f"fc2.weight: {net.fc2.weight.shape}\n\n")
        for i in range(84):
            for j in range(120):
                f.write(f"{net.fc2.weight[i][j]} ")
            f.write("\n")
        f.write("\n")
        f.write(f"fc3.weight: {net.fc3.weight.shape}\n\n")
        for i in range(10):
            for j in range(84):
                f.write(f"{net.fc3.weight[i][j]} ")
            f.write("\n")
        f.write("\n")
        # Save bias of fc
        f.write(f"fc1.bias: {net.fc1.bias.shape}\n\n")
        for i in range(120):
            f.write(f"{net.fc1.bias[i]} ")
        f.write("\n\n")
        f.write(f"fc2.bias: {net.fc2.bias.shape}\n\n")
        for i in range(84):
            f.write(f"{net.fc2.bias[i]} ")
        f.write("\n\n")
        f.write(f"fc3.bias: {net.fc3.bias.shape}\n\n")
        for i in range(10):
            f.write(f"{net.fc3.bias[i]} ")
        f.write("\n\n")
    

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
        'convert': convert
    })