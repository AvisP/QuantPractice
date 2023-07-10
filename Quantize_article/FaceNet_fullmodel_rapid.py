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
from facenet import NN1_BN_FaceNet_2K_160_Quantized

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

def manual_model(model_quant, input_fp32, debug_flag):

    if debug_flag:
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

    input_quant_manual_after_quant, input_quant_manual_after_quant_dequant = quantize_vector(input_fp32, model_quant.quant.scale, model_quant.quant.zero_point)
    if debug_flag:
        print("First Quantized Block_Difference :", torch.sum(torch.dequantize(activations[0]['output']) - input_quant_manual_after_quant_dequant))

    manual_after_batch_norm_quant, manual_after_batch_norm_dequant = batch_norm_quant(input_quant_manual_after_quant_dequant, model_quant.conv1[0])
    if debug_flag:
        print("First batch norm comparison " , torch.sum(torch.dequantize(activations[1]['output']) -  manual_after_batch_norm_dequant))

    manual_after_first_conv_quant, manual_after_first_conv_dequant = convolve_quant(manual_after_batch_norm_dequant[0], model_quant.conv1[1])
    torch.nn.functional.relu(manual_after_first_conv_dequant, inplace=True)
    if debug_flag:
        print("First conv layer comparison ", torch.sum(torch.dequantize(activations[2]['output']) - manual_after_first_conv_dequant.detach().numpy()))

    manual_after_first_pool = model_quant.pool1(manual_after_first_conv_dequant)
    if debug_flag:
        print("First Pooling layer comparison ", torch.sum(torch.dequantize(activations[5]['output']) - manual_after_first_pool.detach().numpy()))

    manual_after_conv2a_quant, manual_after_conv2a_dequant = convolve_quant(manual_after_first_pool, model_quant.conv2a[0])
    if debug_flag:
        print("Conv2a output comparison ", torch.sum(torch.dequantize(activations[6]['output']) - manual_after_conv2a_dequant.detach().numpy()))

    manual_after_conv2_quant, manual_after_conv2_dequant = convolve_quant(manual_after_conv2a_dequant, model_quant.conv2[0])
    if debug_flag:
        print("Conv2 output comparison ", torch.sum(torch.dequantize(activations[9]['output']) - manual_after_conv2_dequant.detach().numpy()))

    manual_after_pool2 = model_quant.pool2(manual_after_conv2_dequant)
    if debug_flag:
        print("Pool2 output comparison ", torch.sum(torch.dequantize(activations[12]['output']) - manual_after_pool2.detach().numpy()))

    manual_after_conv3a_quant, manual_after_conv3a_dequant = convolve_quant(manual_after_pool2, model_quant.conv3a[0])
    if debug_flag:
        print("Conv3a output comparison ", torch.sum(torch.dequantize(activations[13]['output']) - manual_after_conv3a_dequant.detach().numpy()))

    manual_after_conv3_quant, manual_after_conv3_dequant = convolve_quant(manual_after_conv3a_dequant, model_quant.conv3[0])
    if debug_flag:
        print("Conv3 output comparison ", torch.sum(torch.dequantize(activations[16]['output']) - manual_after_conv3_dequant.detach().numpy()))

    manual_after_pool3 = model_quant.pool3(manual_after_conv3_dequant)
    if debug_flag:
        print("Pool3 output comparison ", torch.sum(torch.dequantize(activations[19]['output']) - manual_after_pool3.detach().numpy()))

    manual_after_conv4a_quant, manual_after_conv4a_dequant = convolve_quant(manual_after_pool3, model_quant.conv4a[0])
    if debug_flag:
        print("Conv4a output comparison ", torch.sum(torch.dequantize(activations[20]['output']) - manual_after_conv4a_dequant.detach().numpy()))

    manual_after_conv4_quant, manual_after_conv4_dequant = convolve_quant(manual_after_conv4a_dequant, model_quant.conv4[0])
    if debug_flag:
        print("Conv4 output comparison ", torch.sum(torch.dequantize(activations[23]['output']) - manual_after_conv4_dequant.detach().numpy()))

    manual_after_conv5a_quant, manual_after_conv5a_dequant = convolve_quant(manual_after_conv4_dequant, model_quant.conv5a[0])
    if debug_flag:
        print("Conv5a output comparison ", torch.sum(torch.dequantize(activations[26]['output']) - manual_after_conv5a_dequant.detach().numpy()))

    manual_after_conv5_quant, manual_after_conv5_dequant = convolve_quant(manual_after_conv5a_dequant, model_quant.conv5[0])
    if debug_flag:
        print("Conv5 output comparison ", torch.sum(torch.dequantize(activations[29]['output']) - manual_after_conv5_dequant.detach().numpy()))

    manual_after_conv6a_quant, manual_after_conv6a_dequant = convolve_quant(manual_after_conv5_dequant, model_quant.conv6a[0])
    if debug_flag:
        print("Conv6a output comparison ", torch.sum(torch.dequantize(activations[32]['output']) - manual_after_conv6a_dequant.detach().numpy()))

    manual_after_conv6_quant, manual_after_conv6_dequant = convolve_quant(manual_after_conv6a_dequant, model_quant.conv6[0])
    if debug_flag:
        print("Conv6 output comparison ", torch.sum(torch.dequantize(activations[35]['output']) - manual_after_conv6_dequant.detach().numpy()))

    manual_after_pool4 = model_quant.pool4(manual_after_conv6_dequant)
    if debug_flag:
        print("Pool4 output comparison ", torch.sum(torch.dequantize(activations[38]['output']) - manual_after_pool4.detach().numpy()))

    manual_after_flatten =  torch.flatten(manual_after_pool4)

    manual_after_fc1_quant, manual_after_fc1_dequant = linear_layer_quant(manual_after_flatten, model_quant.fc1[0])
    if debug_flag:
        print("Fc1 output comparison : ", torch.sum(torch.dequantize(activations[39]['output']) - manual_after_fc1_dequant.detach().numpy()))

    torch.nn.functional.relu(manual_after_fc1_dequant, inplace=True)

    manual_after_fc1_dropout = model_quant.fc1[2](manual_after_fc1_dequant)
    if debug_flag:
        print("Fc1 dropout comparison : ", torch.sum(torch.dequantize(activations[40]['output']) - manual_after_fc1_dropout.detach().numpy()))

    manual_after_fc7128_quant, manual_after_fc7128_dequant = linear_layer_quant(manual_after_fc1_dropout, model_quant.fc7128[0])
    if debug_flag:
        print("Fc7128 output comparison : ", torch.sum(torch.dequantize(activations[41]['output']) - manual_after_fc7128_dequant.detach().numpy()))

    manual_final_output = nn.functional.normalize(manual_after_fc7128_dequant.detach(), p=2, dim=0)

    return manual_final_output

trfrm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.ToPILImage(mode="RGB"),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((160,160)),
        torchvision.transforms.ToTensor(),
        ])

# input_fp32 = trfrm(Image.open('/home/avishek/Quantization/data/Avishek_2.jpg'))
# input_fp32 = trfrm(Image.open('/home/avishek/FaceNet_Quant/datasets/Merged-short/0000045/001.jpg'))
# input_fp32 = trfrm(Image.open('/home/avishek/FaceNet_Quant/datasets/Merged-short/625/163737.jpg'))
input_fp32 = trfrm(Image.open('/home/avishek/FaceNet_Quant/datasets/Merged-short/7030/151105.jpg'))
input_fp32 = input_fp32.unsqueeze(0).to('cpu')

model_quant = NN1_BN_FaceNet_2K_160_Quantized()
model_quant.eval()

# Fuse Conv, bn and relu
model_quant.fuse_model()

model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_quant, inplace=True)
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.convert(model_quant, inplace=True)

debug_flag = False

if debug_flag:
    print(model_quant)

checkpoint = torch.load('/home/avishek/Quantization/model_weights/Facenet_2K_160_perchannel_PT_Train_0.854_Test_0.844.pth', map_location=torch.device('cpu'))
model_quant.load_state_dict(checkpoint['state_dict'])

output_quant_model = model_quant(input_fp32)

print('Computing manual model')
manual_output = manual_model(model_quant, input_fp32, debug_flag)

print("Final output comparison :", torch.sum(output_quant_model - manual_output))