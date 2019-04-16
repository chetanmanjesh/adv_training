import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import sys, os, argparse
cuda = torch.device('cuda') 

# functions to show an image
def imshow(img, output):
    img = torchvision.utils.make_grid(img)
    npimg = img.detach().cpu().numpy() / 2.0 + 0.5     # unnormalize
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    plt.savefig(output + '.png')

def cifar10_loader():
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,  testloader, classes


def mnist_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)
    #print(len(trainset))
    #print(trainset[0])
    sys.stdout.flush()

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)
    classes = ('0','1','2','3','4','5','6','7','8','9')
    return trainloader,  testloader, classes


class LinearLayer_CIFAR10(nn.Module):
    def __init__(self):
        super(LinearLayer_CIFAR10, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = self.fc(x.view(-1, 3 * 32 * 32))
        return x

class LinearLayer_MNIST(nn.Module):
    def __init__(self):
        super(LinearLayer_MNIST, self).__init__()
        self.fc = nn.Linear(1 * 28 * 28, 10)

    def forward(self, x):
        x = self.fc(x.view(-1, 1 * 28 * 28))
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_MNIST(models.ResNet):
    def __init__(self, block, layers):
        super(ResNet_MNIST, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

def resnet101_MNIST():
    model = ResNet_MNIST(Bottleneck, [3, 4, 23, 3])
    return model

def train(args, trainloader, testloader):
    if args.data == 'cifar10':
        if args.model == 'resnet':
            model = models.resnet101(pretrained=False).cuda()
        elif args.model == 'linear':
            model = LinearLayer_CIFAR10().cuda()
        else:
            return -1
    elif args.data == 'mnist':
        if args.model == 'resnet':
            model = resnet101_MNIST().cuda()
        elif args.model == 'linear':
            model = LinearLayer_MNIST().cuda()

    try:
        os.makedirs('ckpt_%s_%s' %(args.data, args.model))
    except:
        pass

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    if args.reload == True and os.path.exists('ckpt_%s_%s/ckpt' %(args.data, args.model)):
        print('reload == True')
        state = torch.load('ckpt_%s_%s/ckpt' %(args.data, args.model))
        best_acc = state['best_acc']
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch']
    else:
        best_acc = 0.0
        start_epoch = 0

    for epoch in range(start_epoch, 300):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.cpu().item()
            predicted = outputs.max(1)[1].cpu()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print('Iter %d, step %d/%d. Loss: %.3f | Acc: %.3f (%d/%d)'
                % (epoch, i, len(trainloader), running_loss/(i+1), 100.*float(correct)/total, correct, total), end='\r')
            sys.stdout.flush()
        print('Iter %d. Loss: %.3f | Acc: %.3f (%d/%d)'
                % (epoch, running_loss/len(trainloader), 100.*float(correct)/total, correct, total))
        sys.stdout.flush()
        acc = test(model, testloader)
        scheduler.step(acc)
        test_adversarial(model, criterion, optimizer, testloader)
        if acc > best_acc:
            best_acc = acc
            state = {
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch + 1
            }
            torch.save(state, './ckpt_%s_%s/ckpt' %(args.data, args.model))
        running_loss, correct, total = 0.0, 0, 0

    test_adversarial(model, criterion, optimizer, testloader, output=args.data + '.' + args.model, draw=True)
    print('Finished Training')

def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    print('Test accuracy on the test images: %.3f %%' % (
        100 * float(correct) / total))
    sys.stdout.flush()
    return float(correct) / total

def test_adversarial(model, criterion, optimizer, testloader, output = 'sample', draw=False):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        optimizer.zero_grad()
        images = Variable(images.cuda(), requires_grad=True)
        outputs = model(images)
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        gradient_input = images.grad.data
        #print('gradient_input', gradient_input)
        gradient_input_sign = 2 * (gradient_input > 0).float() - 1.0
        #print('gradient_input_sign', gradient_input_sign)
        optimizer.zero_grad()
        if draw == True:
            imshow(images, output)
            imshow(images + gradient_input_sign * 0.1, output+'.adversarial')
            draw = False
        outputs = model(images + gradient_input_sign * 0.1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()

    print('Adversarial test accuracy on the test images: %.3f %%' % (
        100 * float(correct) / total))
    sys.stdout.flush()
    return float(correct) / total
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='660 Project')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', default='resnet', help='model structure')
    parser.add_argument('--data', default='cifar10', help='dataset: cifar10 or mnist')
    parser.add_argument('--reload', action='store_true', help='reload model')
    args = parser.parse_args()
    print(args)
    # load data
    if args.data == 'cifar10':
        trainloader, testloader, classes = cifar10_loader()
    elif args.data == 'mnist':
        trainloader, testloader, classes = mnist_loader()
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    train(args, trainloader, testloader)
