import torch
from custom_convolve import convolve_torch


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