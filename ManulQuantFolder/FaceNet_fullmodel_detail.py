import numpy as np
import torch
import torch.nn as nn
import torch.quantization
from custom_convolve import convolve_torch, convolve_numpy
torch.set_printoptions(precision=30)
np.set_printoptions(precision=30)
torch.manual_seed(123)

from PIL import Image
import torchvision

class M_quant_fullweight(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):
        super(M_quant_fullweight, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
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

        self.dequant = torch.quantization.DeQuantStub()

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

        x = self.dequant(x)
         
        return x

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

def quantize_vector(input_tensor, scale, zero_point):

    integer_tensor = torch.clamp(torch.round(input_tensor.detach()/scale)+zero_point, min=0)

    floating_point_tensor = (integer_tensor - zero_point)*scale.float()

    return integer_tensor, floating_point_tensor

def batch_norm_quant(input_tensor, model_bn_block):

    norm_tensor = torch.empty([1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]])

    for i in range(input_tensor.shape[1]):
        norm_vect = ((input_tensor[0][i] - model_bn_block.running_mean[i])/torch.sqrt(model_bn_block.running_var[i] + model_bn_block.eps))* model_bn_block.weight[i] +  model_bn_block.bias[i]
        norm_tensor[0,i,:,:] = norm_vect

    batch_norm_integer, batch_norm_float = quantize_vector(norm_tensor, model_bn_block.scale, model_bn_block.zero_point)

    return batch_norm_integer, batch_norm_float

def convolve_quant(input_tensor, model_conv_block):

    quant_conv_weight = torch.dequantize(model_conv_block._weight_bias()[0])

    convolve_manual_float = convolve_torch(input_tensor.detach(), quant_conv_weight.detach(), model_conv_block.bias().detach(), stride=model_conv_block.stride[0])

    conv_integer, conv_float = quantize_vector(convolve_manual_float,  torch.tensor(model_conv_block.scale), model_conv_block.zero_point)

    return conv_integer, conv_float

def linear_layer_quant(input_tensor, model_block):

    manual_linear_layer = torch.matmul(torch.dequantize(model_block._weight_bias()[0]), input_tensor) + model_block._weight_bias()[1].detach()

    linear_integer, linear_float = quantize_vector(manual_linear_layer, torch.tensor(model_block.scale), model_block.zero_point)

    return linear_integer, linear_float

trfrm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.ToPILImage(mode="RGB"),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((160,160)),
        torchvision.transforms.ToTensor(),
        ])

input_fp32 = trfrm(Image.open('/home/avishek/Quantization/data/Avishek_2.jpg'))
input_fp32 = input_fp32.unsqueeze(0).to('cpu')

model_quant = M_quant_fullweight()
model_quant.eval()

# Fuse Conv, bn and relu
model_quant.fuse_model()

model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_quant, inplace=True)
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.convert(model_quant, inplace=True)

print(model_quant)

# checkpoint = torch.load('/home/avishek/Quantization/facenet_second_layer.pt', map_location=torch.device('cpu'))
# model_quant.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load('/home/avishek/Quantization/model_weights/Facenet_2K_160_perchannel_PT_Train_0.854_Test_0.844.pth', map_location=torch.device('cpu'))
model_quant.load_state_dict(checkpoint['state_dict'])

activations = []
def custom_hook(module, input, output):
    info = {
        'module': module,
        'input': input,
        'output': output
    }
    activations.append(info)

for name, module in model_quant.named_modules():
    if len(list(module.children())) == 0:
        print(name, module)
        module.register_forward_hook(custom_hook)

output_quant = model_quant(input_fp32)

input_quant_manual_after_quant, input_quant_manual_after_quant_dequant = quantize_vector(input_fp32, model_quant.quant.scale, model_quant.quant.zero_point)
print("First Quantized Block_Difference :", torch.sum(torch.dequantize(activations[0]['output']) - input_quant_manual_after_quant_dequant))

## For debugging purposes
# print("Running Mean ", model_quant.conv1[0].running_mean)
# print("Running Variance ", model_quant.conv1[0].running_var)
# print("Batch Norm Scale ", model_quant.conv1[0].scale)
# print("Batch Norm Zero Point", model_quant.conv1[0].zero_point)
# print("Batch Norm Eps ", model_quant.conv1[0].eps)
# print("Batch Norm Gamma ", model_quant.conv1[0].weight)
# print("Batch Norm Beta ", model_quant.conv1[0].bias)

manual_after_batch_norm_quant, manual_after_batch_norm_dequant = batch_norm_quant(input_quant_manual_after_quant_dequant, model_quant.conv1[0])
print("First batch norm comparison " , torch.sum(torch.dequantize(activations[1]['output']) -  manual_after_batch_norm_dequant))

manual_after_first_conv_quant, manual_after_first_conv_dequant = convolve_quant(manual_after_batch_norm_dequant[0], model_quant.conv1[1])
torch.nn.functional.relu(manual_after_first_conv_dequant, inplace=True)
print("First conv layer comparison ", torch.sum(torch.dequantize(activations[2]['output']) - manual_after_first_conv_dequant.detach().numpy()))

manual_after_first_pool = model_quant.pool1(manual_after_first_conv_dequant)
print("First Pooling layer comparison ", torch.sum(torch.dequantize(activations[5]['output']) - manual_after_first_pool.detach().numpy()))

manual_after_conv2a_quant, manual_after_conv2a_dequant = convolve_quant(manual_after_first_pool, model_quant.conv2a[0])
print("Conv2a output comparison ", torch.sum(torch.dequantize(activations[6]['output']) - manual_after_conv2a_dequant.detach().numpy()))

manual_after_conv2_quant, manual_after_conv2_dequant = convolve_quant(manual_after_conv2a_dequant, model_quant.conv2[0])
print("Conv2 output comparison ", torch.sum(torch.dequantize(activations[9]['output']) - manual_after_conv2_dequant.detach().numpy()))

manual_after_pool2 = model_quant.pool2(manual_after_conv2_dequant)
print("Pool2 output comparison ", torch.sum(torch.dequantize(activations[12]['output']) - manual_after_pool2.detach().numpy()))

manual_after_conv3a_quant, manual_after_conv3a_dequant = convolve_quant(manual_after_pool2, model_quant.conv3a[0])
print("Conv3a output comparison ", torch.sum(torch.dequantize(activations[13]['output']) - manual_after_conv3a_dequant.detach().numpy()))

manual_after_conv3_quant, manual_after_conv3_dequant = convolve_quant(manual_after_conv3a_dequant, model_quant.conv3[0])
print("Conv3 output comparison ", torch.sum(torch.dequantize(activations[16]['output']) - manual_after_conv3_dequant.detach().numpy()))

manual_after_pool3 = model_quant.pool3(manual_after_conv3_dequant)
print("Pool3 output comparison ", torch.sum(torch.dequantize(activations[19]['output']) - manual_after_pool3.detach().numpy()))

manual_after_conv4a_quant, manual_after_conv4a_dequant = convolve_quant(manual_after_pool3, model_quant.conv4a[0])
print("Conv4a output comparison ", torch.sum(torch.dequantize(activations[20]['output']) - manual_after_conv4a_dequant.detach().numpy()))

manual_after_conv4_quant, manual_after_conv4_dequant = convolve_quant(manual_after_conv4a_dequant, model_quant.conv4[0])
print("Conv4 output comparison ", torch.sum(torch.dequantize(activations[23]['output']) - manual_after_conv4_dequant.detach().numpy()))

manual_after_conv5a_quant, manual_after_conv5a_dequant = convolve_quant(manual_after_conv4_dequant, model_quant.conv5a[0])
print("Conv5a output comparison ", torch.sum(torch.dequantize(activations[26]['output']) - manual_after_conv5a_dequant.detach().numpy()))

manual_after_conv5_quant, manual_after_conv5_dequant = convolve_quant(manual_after_conv5a_dequant, model_quant.conv5[0])
print("Conv5 output comparison ", torch.sum(torch.dequantize(activations[29]['output']) - manual_after_conv5_dequant.detach().numpy()))

manual_after_conv6a_quant, manual_after_conv6a_dequant = convolve_quant(manual_after_conv5_dequant, model_quant.conv6a[0])
print("Conv6a output comparison ", torch.sum(torch.dequantize(activations[32]['output']) - manual_after_conv6a_dequant.detach().numpy()))

manual_after_conv6_quant, manual_after_conv6_dequant = convolve_quant(manual_after_conv6a_dequant, model_quant.conv6[0])
print("Conv6 output comparison ", torch.sum(torch.dequantize(activations[35]['output']) - manual_after_conv6_dequant.detach().numpy()))

manual_after_pool4 = model_quant.pool4(manual_after_conv6_dequant)
print("Pool4 output comparison ", torch.sum(torch.dequantize(activations[38]['output']) - manual_after_pool4.detach().numpy()))

manual_after_flatten =  torch.flatten(manual_after_pool4)
print(manual_after_pool4.shape)

manual_after_fc1_quant, manual_after_fc1_dequant = linear_layer_quant(manual_after_flatten, model_quant.fc1[0])
print("Fc1 output comparison : ", torch.sum(torch.dequantize(activations[39]['output']) - manual_after_fc1_dequant.detach().numpy()))

torch.nn.functional.relu(manual_after_fc1_dequant, inplace=True)

manual_after_fc1_dropout = model_quant.fc1[2](manual_after_fc1_dequant)
print("Fc1 dropout comparison : ", torch.sum(torch.dequantize(activations[40]['output']) - manual_after_fc1_dropout.detach().numpy()))

manual_after_fc7128_quant, manual_after_fc7128_dequant = linear_layer_quant(manual_after_fc1_dropout, model_quant.fc7128[0])
print("Fc7128 output comparison : ", torch.sum(torch.dequantize(activations[41]['output']) - manual_after_fc7128_dequant.detach().numpy()))

print("Final output comparison :", torch.sum(output_quant - manual_after_fc7128_dequant.detach().numpy()))

# model_quant.fc1[0].weight().int_repr()
# torch.dequantize(model_quant.fc1[0].weight()).shape

# if model_quant.fc1[0]._weight_bias()[0].qscheme() is kperChannelAffine then scale and zero_point
# model_quant.fc1[0]._weight_bias()[0].q_per_channel_scales()

# model_quant.fc1[0]._weight_bias()[0].q_per_channel_zero_points()

# # if kPerTensorAffine is true then scale and zero point
# model_quant.fc1[0]._weight_bias()[0].q_scale()

# model_quant.fc1[0]._weight_bias()[0].q_zero_point()

# torch.matmul(torch.dequantize(model_quant.fc1[0].weight()), manual_after_flatten) + model_quant.fc1[0].bias().shape

# torch.matmul(torch.dequantize(model_quant.fc1[0]._weight_bias()[0][0]), manual_after_flatten) + model_quant.fc1[0]._weight_bias()[1][0].detach()

# ans_with_bias = torch.matmul(torch.dequantize(model_quant.fc1[0]._weight_bias()[0][0]), manual_after_flatten) + model_quant.fc1[0]._weight_bias()[1][0].detach()
# quantize_vector(ans_with_bias, torch.tensor(model_quant.fc1[0].scale), model_quant.fc1[0].zero_point)