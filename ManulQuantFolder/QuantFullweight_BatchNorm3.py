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

    def __init__(self):
            super(M_quant_fullweight, self).__init__()
            # QuantStub converts tensors from floating point to quantized
            self.quant = torch.quantization.QuantStub()
            self.conv1 = nn.Sequential(nn.BatchNorm2d(3))
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = self.conv1(x)    
        x = self.dequant(x)

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
         
        return x

def quantize_vector(input_tensor, scale, zero_point):

    integer_tensor = torch.clamp(torch.round(input_tensor.detach()/scale)+zero_point, min=0)

    floating_point_tensor = (integer_tensor - zero_point)*scale.float()

    return integer_tensor, floating_point_tensor

def batch_norm_quant(input_tensor, model_block):

    norm_tensor = torch.empty([1, input_fp32.shape[1], input_fp32.shape[2], input_fp32.shape[3]])

    for i in range(input_fp32.shape[1]):
        norm_vect = ((input_quant_manual_after_quant_dequant[0][i] - model_quant.conv1[0].running_mean[i])/torch.sqrt(model_quant.conv1[0].running_var[i] + model_quant.conv1[0].eps))* model_quant.conv1[0].weight[i] +  model_quant.conv1[0].bias[i]
        norm_tensor[0,i,:,:] = norm_vect

    batch_norm_integer, batch_norm_float = quantize_vector(norm_tensor, model_block.scale, model_block.zero_point)

    return batch_norm_integer, batch_norm_float

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

model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_quant, inplace=True)
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.convert(model_quant, inplace=True)

print(model_quant)

checkpoint = torch.load('/home/avishek/Quantization/BatchNorm_layer.pt', map_location=torch.device('cpu'))
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

# input_quant_manual_after_quant = torch.clamp(torch.round(input_fp32.detach()/model_quant.quant.scale)+model_quant.quant.zero_point, min=0)

# input_quant_manual_after_quant_dequant = (input_quant_manual_after_quant - model_quant.quant.zero_point)*model_quant.quant.scale.float()

input_quant_manual_after_quant, input_quant_manual_after_quant_dequant = quantize_vector(input_fp32, model_quant.quant.scale, model_quant.quant.zero_point)

# Validation quant layer
print("First Quantized Block_Difference :", torch.sum(torch.dequantize(activations[0]['output']) - input_quant_manual_after_quant_dequant))

## For debugging purposes
# print("Running Mean ", model_quant.conv1[0].running_mean)
# print("Running Variance ", model_quant.conv1[0].running_var)
# print("Batch Norm Scale ", model_quant.conv1[0].scale)
# print("Batch Norm Zero Point", model_quant.conv1[0].zero_point)
# print("Batch Norm Eps ", model_quant.conv1[0].eps)
# print("Batch Norm Gamma ", model_quant.conv1[0].weight)
# print("Batch Norm Beta ", model_quant.conv1[0].bias)


print("Input to batch norm ", activations[1]['input'][0][0][0][0][0])
print("Output of batch norm ", activations[1]['output'][0][0][0][0])

# norm_tensor = torch.empty([1, input_fp32.shape[1], input_fp32.shape[2], input_fp32.shape[3]])

# for i in range(input_fp32.shape[1]):
#     norm_vect = ((input_quant_manual_after_quant_dequant[0][i] - model_quant.conv1[0].running_mean[i])/torch.sqrt(model_quant.conv1[0].running_var[i] + model_quant.conv1[0].eps))* model_quant.conv1[0].weight[i] +  model_quant.conv1[0].bias[i]
#     norm_tensor[0,i,:,:] = norm_vect

# manual_after_batch_norm_quant = torch.clamp(torch.round(norm_tensor.detach()/model_quant.conv1[0].scale)+model_quant.conv1[0].zero_point, min=0)

# manual_after_batch_norm_dequant = (manual_after_batch_norm_quant - model_quant.conv1[0].zero_point)*model_quant.conv1[0].scale.float()

manual_after_batch_norm_quant, manual_after_batch_norm_dequant = batch_norm_quant(input_quant_manual_after_quant_dequant, model_quant.conv1[0])

# print("Batch Norm block output comparison ", torch.sum(torch.dequantize(model_fp32_converted_fullweight.BN(model_fp32_converted_fullweight.quant(input_fp32))) - manual_after_batch_norm_dequant))
# print(output_quant[0][0][0])
# print(manual_after_batch_norm_dequant[0][0][0])
print("First batch norm comparison " , torch.sum(output_quant-  manual_after_batch_norm_dequant))

quant_conv_weight = torch.dequantize(model_quant.conv1[1]._weight_bias()[0])

convolve_manual_float = convolve_torch(input_quant_manual_after_quant_dequant[0].detach(), quant_conv_weight.detach(), model_quant.conv.bias().detach())

manual_after_first_conv_quant, manual_after_first_conv_dequant = quantize_vector(convolve_manual_float,  model_quant.conv1[1].scale, model_quant.conv1[1].zero_point)

print("First conv layer comparison ", torch.sum(output_manual - output_quant[0].detach().numpy()))


