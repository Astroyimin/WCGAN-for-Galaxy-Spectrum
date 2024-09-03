import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Conv1d(in_planes,in_planes//ratio, 1, bias=False),
                            nn.ReLU(),
                            nn.Conv1d(in_planes//ratio,in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes,cbam, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)

        self.ca = ChannelAttention(planes*4)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = cbam

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        if self.cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BottleneckBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, cbam=False,stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)

        self.ca = ChannelAttention(planes*4)
        self.sa = SpatialAttention()


        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.cbam = cbam

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,cbam = False,linear=True):
        super(ResNet, self).__init__()
        self.channels = 64
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(512*block.expansion,num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, layers[0],cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], cbam,stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cbam,stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cbam,stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*block.expansion,num_classes)
        self.Linear = linear

    def _make_layer(self, block, out_channels, blocks, cbam,stride=1):
        downsample = None
        if stride != 1 or self.channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.channels, out_channels,cbam,stride=stride, downsample=downsample))
        self.channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.channels, out_channels,cbam))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.Linear:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else :
            x = self.conv2(x)
            x = self.avgpool(x)
        return x
def restnet18cbam(numberclass,cbam = False,linear=True):
    return ResNet(BottleneckBlock, [2,2,2,2], num_classes=numberclass,cbam = cbam,linear=linear)
def restnet50cbam(numberclass,cbam = False,linear=True):
    return ResNet(ResidualBlock, [3,4,6,3], num_classes=numberclass,cbam = cbam,linear=linear)