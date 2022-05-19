import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm as tqdm
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, args, writer, block, num_blocks, num_classes = 10, use_norm = False):
        super(ResNet_s, self).__init__()
        self.args = args
        self.writer = writer

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def fit(self, train_loader, criterion, optimizer, scheduler, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            for images, labels in train_loader:
                with torch.enable_grad():
                    self.train()
                    images = images.cuda(self.args.gpu)
                    labels = labels.cuda(self.args.gpu)
                    outputs = self.forward(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def evaluate(self, test_loader):
        accuracy = 0
        preds_ = list()
        labels_ = list()
        for images, labels in test_loader:
            with torch.no_grad():
                self.eval()
                images = images.cuda(self.args.gpu)
                labels = labels.cuda(self.args.gpu)
                outputs = self.forward(images)
                _, preds = outputs.max(dim = 1)
                accuracy += (preds == labels).float().sum()
                preds_.extend(preds.tolist())
                labels_.extend(labels.tolist())
        accuracy /= len(test_loader.dataset)
        confusion_matrix_ = confusion_matrix(labels_, preds_)
        return accuracy, confusion_matrix_

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])

def resnet32(args, writer):
    return ResNet_s(args, writer, BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])
