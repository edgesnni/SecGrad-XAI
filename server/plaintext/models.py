import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 


class AliceNet(nn.Module):
    def __init__(self, args, num_classes):
        super(AliceNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
 
    def forward(self, x):
        out = torch.flatten(x, 1) 
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out
        
class AlexNet(nn.Module):
    def __init__(self, args, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        safe_eps = 10000 # 1e-3
        #late_eps = 10000

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64, eps=safe_eps) # Added
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192, eps=safe_eps) # Added
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384, eps=safe_eps) # Added
        self.relu3 = nn.ReLU()

        # Block 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256, eps=safe_eps) # Added
        self.relu4 = nn.ReLU()
        
        # Block 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, eps=safe_eps) # Added
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) 

        # Classifier
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096, eps=safe_eps) # Added for stability
        self.fc_relu1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096, eps=safe_eps) # Added for stability
        self.fc_relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Features
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout1(x)
        x = self.fc_relu1(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = self.fc_relu2(self.bn_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        return x    
        
class VGG11(nn.Module):
    """
    VGG-11 architecture with Batch Normalization (VGG11-BN)
    Explicitly implemented for clarity and CrypTen compatibility.
    """
    def __init__(self, args, num_classes=200, image_channels=3):
        super(VGG11, self).__init__()    

        self.conv1_1 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) 
        self.act1_1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128) 
        self.act2_1 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256) 
        self.act3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256) 
        self.act3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512) 
        self.act4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.act4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.act5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512) 
        self.act5_2 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.act_fc1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.act_fc2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        
        x = self.conv1_1(x)
        x = self.bn1_1(x) 
        x = self.act1_1(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.act2_1(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x) 
        x = self.act3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x) 
        x = self.act3_2(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.bn4_1(x) 
        x = self.act4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x) 
        x = self.act4_2(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x) 
        x = self.act5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x) 
        x = self.act5_2(x)
        x = self.pool5(x)
        
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        
        x = self.drop1(self.act_fc1(self.fc1(x)))
        x = self.drop2(self.act_fc2(self.fc2(x)))
        x = self.fc3(x)
        
        return x
        
class VGG19(nn.Module):
    def __init__(self, args, num_classes=1000, image_channels=3):
        super(VGG19, self).__init__()
        
        # Helper to define layers without Sequential
        self.conv1_1 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1); self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1); self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1); self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1); self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1); self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_4 = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1); self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_4 = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.pool(self.relu(self.bn1_2(self.conv1_2(x))))
        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool(self.relu(self.bn2_2(self.conv2_2(x))))
        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool(self.relu(self.bn3_4(self.conv3_4(x))))
        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool(self.relu(self.bn4_4(self.conv4_4(x))))
        # Block 5
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool(self.relu(self.bn5_4(self.conv5_4(x))))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class InceptionModule(nn.Module):
    """
    GoogLeNet Inception Module (Dimension Reduction version) implemented 
    explicitly without loops or Sequential containers.
    """
    def __init__(self, in_channels, n1x1, n3x3_red, n3x3, n5x5_red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        self.branch1_1x1 = nn.Conv2d(in_channels, n1x1, kernel_size=1)
        self.act1 = nn.ReLU()

        self.branch2_1x1 = nn.Conv2d(in_channels, n3x3_red, kernel_size=1)
        self.act2_1 = nn.ReLU()
        self.branch2_3x3 = nn.Conv2d(n3x3_red, n3x3, kernel_size=3, padding=1)
        self.act2_2 = nn.ReLU()

        self.branch3_1x1 = nn.Conv2d(in_channels, n5x5_red, kernel_size=1)
        self.act3_1 = nn.ReLU()
        self.branch3_5x5 = nn.Conv2d(n5x5_red, n5x5, kernel_size=5, padding=2)
        self.act3_2 = nn.ReLU()

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        self.act4 = nn.ReLU()

    def forward(self, x):

        out1 = self.act1(self.branch1_1x1(x))
        out2 = self.act2_1(self.branch2_1x1(x))
        out2 = self.act2_2(self.branch2_3x3(out2))
        out3 = self.act3_1(self.branch3_1x1(x))
        out3 = self.act3_2(self.branch3_5x5(out3))
        out4 = self.branch4_pool(x)
        out4 = self.act4(self.branch4_1x1(out4))
        
        return torch.cat([out1, out2, out3, out4], dim=1)

class GoogLeNet(nn.Module):
    """
    The full GoogLeNet (Inception v1) architecture using explicit layer
    definitions and Secure Inception Modules. Auxiliary classifiers omitted.
    """
    def __init__(self, args, num_classes=10, image_channels=3, input_size=224):
        super(GoogLeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_red = nn.Conv2d(64, 64, kernel_size=1)
        self.act2_red = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        final_spatial_size = input_size // 32 
        
        self.fc = nn.Linear(1024, num_classes) 

    def forward(self, x):
        
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2_red(self.conv2_red(x))
        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = x.mean(dim=(2, 3)) 
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ConvDownsampler(nn.Module):
    """
    Helper module for ResNet-style downsampling/upsampling in the residual path.
    Performs a 1x1 convolution to match channel count.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        return self.conv(x)
    
class _Downsampler(nn.Module):
    """
    A minimal traceable module to combine the Conv1x1 and BatchNorm 
    required for the residual shortcut when stride != 1 or channels change.
    Used to avoid nn.Sequential or nn.ModuleList in the main class.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block (from the user's previous code).
    It accepts a single nn.Module (like _Downsampler) or None for the shortcut.
    """
    expansion = 1

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample 

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x) 
        out = out + shortcut
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Block used in ResNet-50 and deeper models.
    It uses a 1x1, 3x3, and final 1x1 convolution sequence.
    The output channels are 4x the input planes (planes * expansion).
    """
    expansion = 4 

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) 

        if self.downsample is not None:
            shortcut = self.downsample(x) 

        out = out + shortcut
        out = self.relu(out)
        return out

class CustomResNet18(nn.Module):
    """
    Custom ResNet-18 architecture built entirely by manually naming all 
    modules (8 BasicBlocks, 3 Downsamplers, plus initial/final layers) 
    to ensure full PyTorch traceability without ModuleList or Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(CustomResNet18, self).__init__()
        inplanes = 64
        expansion = 1
        in_channels = 3
        
        self.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1_block1 = BasicBlock(inplanes=64, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block2 = BasicBlock(inplanes=64, planes=64, args=args, stride=1, downsample=None)
        self.layer2_ds = _Downsampler(in_channels=64*expansion, out_channels=128*expansion, stride=2)
        self.layer2_block1 = BasicBlock(inplanes=64, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = BasicBlock(inplanes=128, planes=128, args=args, stride=1, downsample=None)
        self.layer3_ds = _Downsampler(in_channels=128*expansion, out_channels=256*expansion, stride=2)
        self.layer3_block1 = BasicBlock(inplanes=128, planes=256, args=args, stride=2, downsample=self.layer3_ds)
        self.layer3_block2 = BasicBlock(inplanes=256, planes=256, args=args, stride=1, downsample=None)
        self.layer4_ds = _Downsampler(in_channels=256*expansion, out_channels=512*expansion, stride=2)
        self.layer4_block1 = BasicBlock(inplanes=256, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = BasicBlock(inplanes=512, planes=512, args=args, stride=1, downsample=None)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CustomResNet50(nn.Module):
    """
    Custom ResNet-50 architecture using the Bottleneck block.
    Layer structure: [3, 4, 6, 3] Bottleneck blocks.
    All modules are manually named for full traceability without ModuleList/Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(CustomResNet50, self).__init__()
        inplanes = 64
        expansion = 4
        block = Bottleneck
        
        in_channels = 3
        
        self.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_ds = _Downsampler(in_channels=64, out_channels=256, stride=1)
        self.layer1_block1 = block(inplanes=64, planes=64, args=args, stride=1, downsample=self.layer1_ds)
        self.layer1_block2 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block3 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        
        self.layer2_ds = _Downsampler(in_channels=256, out_channels=512, stride=2)
        self.layer2_block1 = block(inplanes=256, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block3 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block4 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)

        self.layer3_ds = _Downsampler(in_channels=512, out_channels=1024, stride=2)
        self.layer3_block1 = block(inplanes=512, planes=256, args=args, stride=2, downsample=self.layer3_ds)
        self.layer3_block2 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block3 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block4 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block5 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block6 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)

        self.layer4_ds = _Downsampler(in_channels=1024, out_channels=2048, stride=2)
        self.layer4_block1 = block(inplanes=1024, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.layer4_block3 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer1_block3(x)

        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        x = self.layer2_block4(x)

        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        x = self.layer3_block4(x)
        x = self.layer3_block5(x)
        x = self.layer3_block6(x)

        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.layer4_block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CustomResNet152(nn.Module):
    """
    Custom ResNet-152 architecture using the Bottleneck block.
    Layer structure: [3, 8, 36, 3] Bottleneck blocks.
    All modules are manually named for full traceability without ModuleList/Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(CustomResNet152, self).__init__()
        inplanes = 64
        expansion = 4
        block = Bottleneck
        
        in_channels = 3
        
        self.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_ds = _Downsampler(in_channels=64, out_channels=256, stride=1)
        self.layer1_block1 = block(inplanes=64, planes=64, args=args, stride=1, downsample=self.layer1_ds)
        self.layer1_block2 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block3 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        
        self.layer2_ds = _Downsampler(in_channels=256, out_channels=512, stride=2)
        self.layer2_block1 = block(inplanes=256, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block3 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block4 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block5 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block6 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block7 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block8 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)

        self.layer3_ds = _Downsampler(in_channels=512, out_channels=1024, stride=2)
        self.layer3_block1 = block(inplanes=512, planes=256, args=args, stride=2, downsample=self.layer3_ds)
        self.layer3_block2 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block3 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block4 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block5 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block6 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block7 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block8 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block9 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block10 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block11 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block12 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block13 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block14 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block15 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block16 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block17 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block18 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block19 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block20 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block21 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block22 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block23 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block24 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block25 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block26 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block27 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block28 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block29 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block30 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block31 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block32 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block33 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block34 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block35 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block36 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        
        self.layer4_ds = _Downsampler(in_channels=1024, out_channels=2048, stride=2)
        self.layer4_block1 = block(inplanes=1024, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.layer4_block3 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer1_block3(x)

        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        
        x = self.layer2_block4(x)
        x = self.layer2_block5(x)
        x = self.layer2_block6(x)
        x = self.layer2_block7(x)
        x = self.layer2_block8(x)

        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        x = self.layer3_block4(x)
        x = self.layer3_block5(x)
        x = self.layer3_block6(x)
        x = self.layer3_block7(x)
        x = self.layer3_block8(x)
        x = self.layer3_block9(x)
        x = self.layer3_block10(x)
        x = self.layer3_block11(x)
        x = self.layer3_block12(x)
        x = self.layer3_block13(x)
        x = self.layer3_block14(x)
        x = self.layer3_block15(x)
        x = self.layer3_block16(x)
        x = self.layer3_block17(x)
        x = self.layer3_block18(x)
        x = self.layer3_block19(x)
        x = self.layer3_block20(x)
        x = self.layer3_block21(x)
        x = self.layer3_block22(x)
        x = self.layer3_block23(x)
        x = self.layer3_block24(x)
        x = self.layer3_block25(x)
        x = self.layer3_block26(x)
        x = self.layer3_block27(x)
        x = self.layer3_block28(x)
        x = self.layer3_block29(x)
        x = self.layer3_block30(x)
        x = self.layer3_block31(x)
        x = self.layer3_block32(x)
        x = self.layer3_block33(x)
        x = self.layer3_block34(x)
        x = self.layer3_block35(x)
        x = self.layer3_block36(x)

        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.layer4_block3(x)
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x
    