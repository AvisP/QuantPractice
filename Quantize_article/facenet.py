import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = dict(
    acc_920='https://github.com/khrlimam/facenet/releases/download/acc-0.920/model920-6be7e3e9.pth',
    acc_921='https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'
)

def load_state(arch, progress=True):
    state = load_state_dict_from_url(model_urls.get(arch), progress=progress)
    return state


def model_920(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_920', progress)
        model.load_state_dict(state['state_dict'])
    return model


def model_921(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_921', progress)
        model.load_state_dict(state['state_dict'])
    return model


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class FaceNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super(FaceNetModel, self).__init__()

        self.model = resnet50(pretrained)
        embedding_size = 128
        num_classes = 500
        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        # modify fc layer based on https://arxiv.org/abs/1703.07737
        self.model.fc = nn.Sequential(
            Flatten(),
            # nn.Linear(100352, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(100352, embedding_size))

        self.model.classifier = nn.Linear(embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    # returns face embedding(embedding_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res










# Taken from: /home/beyza/FaceNet/facenet-v3
class NN1_BN_FaceNet_4K(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_4K, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # ORIGINAL
        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 32*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(32*128, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
 
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features











# Taken from: /home/beyza/FaceNet/facenet-v3-Merged-Real
class NN1_BN_FaceNet_4K_M(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_4K_M, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 32*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(32*128, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-Merged-Real
class NN1_BN_FaceNet_4K_M_Quantized(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_4K_M_Quantized, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 32*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(32*128, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = self.dequant(x)
        
        x = nn.functional.normalize(x, p=2, dim=1)        
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-2048-new
class NN1_BN_FaceNet_2K(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_2K, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # ORIGINAL
        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 2048), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(2048, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')



# Taken from: /home/beyza/FaceNet/facenet-v3-2048-new
class NN1_BN_FaceNet_2K_Quantized(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_2K_Quantized, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # ORIGINAL
        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 2048), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(2048, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = self.dequant(x)

        x = nn.functional.normalize(x, p=2, dim=1)       
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')

# Taken from: /home/beyza/FaceNet/facenet-v3-1024
class NN1_BN_FaceNet_1K(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_1K, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 1024), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(1024, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc7128(x)

        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-1024
class NN1_BN_FaceNet_1K_Quantized(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_1K_Quantized, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 1024), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(1024, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)
        

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc7128(x)
        x = self.dequant(x)

        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-Merged-512
class NN1_BN_FaceNet_05K(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_05K, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 4*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(4*128, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-Merged-512
class NN1_BN_FaceNet_05K_Quantized(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_05K_Quantized, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 4*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(4*128, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = self.dequant(x)

        x = nn.functional.normalize(x, p=2, dim=1)        
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


# Taken from: /home/beyza/FaceNet/facenet-v3-2048-new
class NN1_BN_FaceNet_2K_160(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_2K_160, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Modified for 160 by 160
        self.fc1 = nn.Sequential(nn.Linear(256*5*5, 2048), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 5 5
        self.fc7128 = nn.Sequential(nn.Linear(2048, embedding_size))

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')


class NN1_BN_FaceNet_2K_160_Quantized(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_2K_160_Quantized, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2a = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv3a = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4a = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.conv5a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv6a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # ORIGINAL
        self.fc1 = nn.Sequential(nn.Linear(256*5*5, 16*128), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(16*128, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
            self.to(device)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4(x)

        x = self.conv5a(x)
        x = self.conv5(x)

        x = self.conv6a(x)
        x = self.conv6(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc7128(x)
        
        x = self.dequant(x)
        x = nn.functional.normalize(x, p=2, dim=1)                
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        return features

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')