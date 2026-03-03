import crypten
import crypten.nn as cnn
import numpy as np

class SecureVGG11(cnn.Module):
    """
    VGG-11 architecture with Batch Normalization (VGG11-BN)
    Explicitly implemented for clarity and CrypTen compatibility.
    """
    def __init__(self, args, num_classes=0, image_channels=3):
        super(SecureVGG11, self).__init__()    

        self.conv1_1 = cnn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = cnn.BatchNorm2d(64) 
        self.act1_1 = cnn.ReLU()
        self.pool1 = cnn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2_1 = cnn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = cnn.BatchNorm2d(128) 
        self.act2_1 = cnn.ReLU()
        self.pool2 = cnn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv3_1 = cnn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = cnn.BatchNorm2d(256) 
        self.act3_1 = cnn.ReLU()
        self.conv3_2 = cnn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = cnn.BatchNorm2d(256) 
        self.act3_2 = cnn.ReLU()
        self.pool3 = cnn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv4_1 = cnn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = cnn.BatchNorm2d(512) 
        self.act4_1 = cnn.ReLU()
        self.conv4_2 = cnn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = cnn.BatchNorm2d(512)
        self.act4_2 = cnn.ReLU()
        self.pool4 = cnn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv5_1 = cnn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = cnn.BatchNorm2d(512)
        self.act5_1 = cnn.ReLU()
        self.conv5_2 = cnn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = cnn.BatchNorm2d(512) 
        self.act5_2 = cnn.ReLU()
        self.pool5 = cnn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = cnn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = cnn.Linear(512 * 7 * 7, 4096)
        self.act_fc1 = cnn.ReLU()
        self.drop1 = cnn.Dropout(p=0.5)

        self.fc2 = cnn.Linear(4096, 4096)
        self.act_fc2 = cnn.ReLU()
        self.drop2 = cnn.Dropout(p=0.5)

        self.fc3 = cnn.Linear(4096, num_classes)

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
        x = x.flatten(1)
        
        x = self.drop1(self.act_fc1(self.fc1(x)))
        x = self.drop2(self.act_fc2(self.fc2(x)))
        x = self.fc3(x)
        
        return x
        
class SecureVGG19(cnn.Module):
    def __init__(self, args, num_classes=0, image_channels=3):
        super(SecureVGG19, self).__init__()
        
        # Helper to define layers without Sequential
        self.conv1_1 = cnn.Conv2d(image_channels, 64, kernel_size=3, padding=1); self.bn1_1 = cnn.BatchNorm2d(64)
        self.conv1_2 = cnn.Conv2d(64, 64, kernel_size=3, padding=1); self.bn1_2 = cnn.BatchNorm2d(64)
        
        self.conv2_1 = cnn.Conv2d(64, 128, kernel_size=3, padding=1); self.bn2_1 = cnn.BatchNorm2d(128)
        self.conv2_2 = cnn.Conv2d(128, 128, kernel_size=3, padding=1); self.bn2_2 = cnn.BatchNorm2d(128)
        
        self.conv3_1 = cnn.Conv2d(128, 256, kernel_size=3, padding=1); self.bn3_1 = cnn.BatchNorm2d(256)
        self.conv3_2 = cnn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_2 = cnn.BatchNorm2d(256)
        self.conv3_3 = cnn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_3 = cnn.BatchNorm2d(256)
        self.conv3_4 = cnn.Conv2d(256, 256, kernel_size=3, padding=1); self.bn3_4 = cnn.BatchNorm2d(256)
        
        self.conv4_1 = cnn.Conv2d(256, 512, kernel_size=3, padding=1); self.bn4_1 = cnn.BatchNorm2d(512)
        self.conv4_2 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_2 = cnn.BatchNorm2d(512)
        self.conv4_3 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_3 = cnn.BatchNorm2d(512)
        self.conv4_4 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn4_4 = cnn.BatchNorm2d(512)
        
        self.conv5_1 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_1 = cnn.BatchNorm2d(512)
        self.conv5_2 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_2 = cnn.BatchNorm2d(512)
        self.conv5_3 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_3 = cnn.BatchNorm2d(512)
        self.conv5_4 = cnn.Conv2d(512, 512, kernel_size=3, padding=1); self.bn5_4 = cnn.BatchNorm2d(512)

        self.relu = cnn.ReLU()
        self.pool = cnn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = cnn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = cnn.Linear(512 * 7 * 7, 4096)
        self.drop1 = cnn.Dropout(0.5)
        self.fc2 = cnn.Linear(4096, 4096)
        self.drop2 = cnn.Dropout(0.5)
        self.fc3 = cnn.Linear(4096, num_classes)
    
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

        x = x.flatten(1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class SecureAlexNet(cnn.Module):
    def __init__(self, args, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        # Block 1
        self.conv1 = cnn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = cnn.BatchNorm2d(64) # Added
        self.relu1 = cnn.ReLU()
        self.pool1 = cnn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 2
        self.conv2 = cnn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = cnn.BatchNorm2d(192) # Added
        self.relu2 = cnn.ReLU()
        self.pool2 = cnn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 3
        self.conv3 = cnn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = cnn.BatchNorm2d(384) # Added
        self.relu3 = cnn.ReLU()

        # Block 4
        self.conv4 = cnn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = cnn.BatchNorm2d(256) # Added
        self.relu4 = cnn.ReLU()
        
        # Block 5
        self.conv5 = cnn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = cnn.BatchNorm2d(256) # Added
        self.relu5 = cnn.ReLU()
        self.pool5 = cnn.MaxPool2d(kernel_size=3, stride=2)
        
        self.avgpool = cnn.AdaptiveAvgPool2d((6, 6)) 

        # Classifier
        self.dropout1 = cnn.Dropout(p=dropout)
        self.fc1 = cnn.Linear(256 * 6 * 6, 4096)
        self.bn_fc1 = cnn.BatchNorm1d(4096) # Added for stability
        self.fc_relu1 = cnn.ReLU()
        
        self.dropout2 = cnn.Dropout(p=dropout)
        self.fc2 = cnn.Linear(4096, 4096)
        self.bn_fc2 = cnn.BatchNorm1d(4096) # Added for stability
        self.fc_relu2 = cnn.ReLU()
        
        self.fc3 = cnn.Linear(4096, num_classes)
    
    def forward(self, x):
        # Features
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        
        x = self.avgpool(x)
        x = x.flatten(1)
        
        # Classifier
        x = self.dropout1(x)
        x = self.fc_relu1(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = self.fc_relu2(self.bn_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        return x    
        



class InceptionModule(cnn.Module):
    """
    GoogLeNet Inception Module (Dimension Reduction version) implemented 
    explicitly without loops or Sequential containers.
    """
    def __init__(self, in_channels, n1x1, n3x3_red, n3x3, n5x5_red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        self.branch1_1x1 = cnn.Conv2d(in_channels, n1x1, kernel_size=1)
        self.act1 = cnn.ReLU()

        self.branch2_1x1 = cnn.Conv2d(in_channels, n3x3_red, kernel_size=1)
        self.act2_1 = cnn.ReLU()
        self.branch2_3x3 = cnn.Conv2d(n3x3_red, n3x3, kernel_size=3, padding=1)
        self.act2_2 = cnn.ReLU()

        self.branch3_1x1 = cnn.Conv2d(in_channels, n5x5_red, kernel_size=1)
        self.act3_1 = cnn.ReLU()
        self.branch3_5x5 = cnn.Conv2d(n5x5_red, n5x5, kernel_size=5, padding=2)
        self.act3_2 = cnn.ReLU()

        self.branch4_pool = cnn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = cnn.Conv2d(in_channels, pool_proj, kernel_size=1)
        self.act4 = cnn.ReLU()

    def forward(self, x):
        out1 = self.act1(self.branch1_1x1(x))
        out2 = self.act2_1(self.branch2_1x1(x))
        out2 = self.act2_2(self.branch2_3x3(out2))
        out3 = self.act3_1(self.branch3_1x1(x))
        out3 = self.act3_2(self.branch3_5x5(out3))
        out4 = self.branch4_pool(x)
        out4 = self.act4(self.branch4_1x1(out4))

        return crypten.cat([out1, out2, out3, out4], dim=1) #out1.cat([out2, out3, out4], dim=1) # cnn.cat([out1, out2, out3, out4], dim=1)

class SecureGoogLeNet(cnn.Module):
    """
    The full GoogLeNet (Inception v1) architecture using explicit layer
    definitions and Secure Inception Modules. Auxiliary classifiers omitted.
    """
    def __init__(self, args, num_classes=10, image_channels=3, input_size=224):
        super(SecureGoogLeNet, self).__init__()
        
        self.conv1 = cnn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.act1 = cnn.ReLU()
        self.pool1 = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_red = cnn.Conv2d(64, 64, kernel_size=1)
        self.act2_red = cnn.ReLU()
        self.conv2 = cnn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.act2 = cnn.ReLU()
        self.pool2 = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        final_spatial_size = input_size // 32 
        self.fc = cnn.Linear(1024, num_classes) 

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
        x = x.flatten(1)
        x = self.fc(x)

        return x


class SecureConvDownsampler(cnn.Module):
    """
    Helper module for ResNet-style downsampling/upsampling in the residual path.
    Performs a 1x1 convolution to match channel count.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = cnn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        return self.conv(x)
    
class _SecureDownsampler(cnn.Module):
    """
    A minimal traceable module to combine the Conv1x1 and BatchNorm 
    required for the residual shortcut when stride != 1 or channels change.
    Used to avoid cnn.Sequential or cnn.ModuleList in the main class.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = cnn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = cnn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SecureBasicBlock(cnn.Module):
    """
    Standard ResNet Basic Block (from the user's previous code).
    It accepts a single cnn.Module (like _SecureDownsampler) or None for the shortcut.
    """
    expansion = 1

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(SecureBasicBlock, self).__init__()

        self.conv1 = cnn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = cnn.BatchNorm2d(planes)
        self.relu = cnn.ReLU()
        
        self.conv2 = cnn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = cnn.BatchNorm2d(planes)
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

class SecureBottleneckBlock(cnn.Module):
    """
    ResNet Bottleneck Block (1x1 -> 3x3 -> 1x1 convolutions)
    """
    expansion = 4 

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(SecureBottleneckBlock, self).__init__()
        
        self.conv1 = cnn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = cnn.BatchNorm2d(planes)
        self.relu = cnn.ReLU()
        
        self.conv2 = cnn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = cnn.BatchNorm2d(planes)

        self.conv3 = cnn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = cnn.BatchNorm2d(planes * self.expansion)
        
        self.downsample = downsample 
        self.stride = stride

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

class SecureCustomResNet18(cnn.Module):
    """
    Custom ResNet-18 architecture built entirely by manually naming all 
    modules (8 SecureBasicBlocks, 3 Downsamplers, plus initial/final layers) 
    to ensure full PyTorch traceability without ModuleList or Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(SecureCustomResNet18, self).__init__()
        inplanes = 64
        expansion = 1
        in_channels = 3
        
        self.conv1 = cnn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = cnn.BatchNorm2d(inplanes)
        self.relu = cnn.ReLU()
        self.maxpool = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_block1 = SecureBasicBlock(inplanes=64, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block2 = SecureBasicBlock(inplanes=64, planes=64, args=args, stride=1, downsample=None)
        
        self.layer2_ds = _SecureDownsampler(in_channels=64*expansion, out_channels=128*expansion, stride=2)
        self.layer2_block1 = SecureBasicBlock(inplanes=64, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = SecureBasicBlock(inplanes=128, planes=128, args=args, stride=1, downsample=None)
        
        self.layer3_ds = _SecureDownsampler(in_channels=128*expansion, out_channels=256*expansion, stride=2)
        self.layer3_block1 = SecureBasicBlock(inplanes=128, planes=256, args=args, stride=2, downsample=self.layer3_ds)
        self.layer3_block2 = SecureBasicBlock(inplanes=256, planes=256, args=args, stride=1, downsample=None)

        self.layer4_ds = _SecureDownsampler(in_channels=256*expansion, out_channels=512*expansion, stride=2)
        self.layer4_block1 = SecureBasicBlock(inplanes=256, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = SecureBasicBlock(inplanes=512, planes=512, args=args, stride=1, downsample=None)

        self.avgpool = cnn.AdaptiveAvgPool2d((1, 1))
        self.fc = cnn.Linear(512 * expansion, num_classes)
        
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
        #x = x.flatten(1)
        x = self.fc(x)
        return x

class SecureCustomResNet50(cnn.Module):
    """
    Custom ResNet-50 architecture using the Bottleneck block.
    Layer structure: [3, 4, 6, 3] Bottleneck blocks.
    All modules are manually named for full traceability without ModuleList/Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(SecureCustomResNet50, self).__init__()
        inplanes = 64
        expansion = 4
        block = SecureBottleneckBlock

        in_channels = 3
        
        self.conv1 = cnn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = cnn.BatchNorm2d(inplanes)
        self.relu = cnn.ReLU()
        self.maxpool = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_ds = _SecureDownsampler(in_channels=64, out_channels=256, stride=1)
        self.layer1_block1 = block(inplanes=64, planes=64, args=args, stride=1, downsample=self.layer1_ds)
        self.layer1_block2 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block3 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        
        self.layer2_ds = _SecureDownsampler(in_channels=256, out_channels=512, stride=2)
        self.layer2_block1 = block(inplanes=256, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block3 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block4 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)

        self.layer3_ds = _SecureDownsampler(in_channels=512, out_channels=1024, stride=2)
        self.layer3_block1 = block(inplanes=512, planes=256, args=args, stride=2, downsample=self.layer3_ds)
        self.layer3_block2 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block3 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block4 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block5 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)
        self.layer3_block6 = block(inplanes=1024, planes=256, args=args, stride=1, downsample=None)

        self.layer4_ds = _SecureDownsampler(in_channels=1024, out_channels=2048, stride=2)
        self.layer4_block1 = block(inplanes=1024, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.layer4_block3 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.avgpool = cnn.AdaptiveAvgPool2d((1, 1))
        self.fc = cnn.Linear(512 * expansion, num_classes) 
        
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
        x = x.flatten(1) 
        x = self.fc(x)
        return x

class SecureCustomResNet152(cnn.Module):
    """
    Custom ResNet-152 architecture using the Bottleneck block.
    Layer structure: [3, 8, 36, 3] Bottleneck blocks.
    All modules are manually named for full traceability without ModuleList/Sequential.
    """
    def __init__(self, args, num_classes=10):
        super(SecureCustomResNet152, self).__init__()
        inplanes = 64
        expansion = 4
        block = SecureBottleneckBlock
        
        in_channels = 3
        
        self.conv1 = cnn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = cnn.BatchNorm2d(inplanes)
        self.relu = cnn.ReLU()
        self.maxpool = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_ds = _SecureDownsampler(in_channels=64, out_channels=256, stride=1)
        self.layer1_block1 = block(inplanes=64, planes=64, args=args, stride=1, downsample=self.layer1_ds)
        self.layer1_block2 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        self.layer1_block3 = block(inplanes=256, planes=64, args=args, stride=1, downsample=None)
        
        self.layer2_ds = _SecureDownsampler(in_channels=256, out_channels=512, stride=2)
        self.layer2_block1 = block(inplanes=256, planes=128, args=args, stride=2, downsample=self.layer2_ds)
        self.layer2_block2 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block3 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block4 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block5 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block6 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block7 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)
        self.layer2_block8 = block(inplanes=512, planes=128, args=args, stride=1, downsample=None)

        self.layer3_ds = _SecureDownsampler(in_channels=512, out_channels=1024, stride=2)
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
        
        self.layer4_ds = _SecureDownsampler(in_channels=1024, out_channels=2048, stride=2)
        self.layer4_block1 = block(inplanes=1024, planes=512, args=args, stride=2, downsample=self.layer4_ds)
        self.layer4_block2 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)
        self.layer4_block3 = block(inplanes=2048, planes=512, args=args, stride=1, downsample=None)

        self.avgpool = cnn.AdaptiveAvgPool2d((1, 1))
        self.fc = cnn.Linear(512 * expansion, num_classes)
        
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
        x = x = x.flatten(1) 
        x = self.fc(x)
        return x