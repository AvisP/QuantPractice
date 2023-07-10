import numpy as np
import torch
import torch.nn as nn
import sys
from custom_convolve import convolve_torch, convolve_numpy
torch.set_printoptions(precision=40)
np.set_printoptions(precision=30)
# torch.manual_seed(159)
# np.random.seed(129)

class M2_double(nn.Module):

    def __init__(self):
        super(M2_double, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        # self.BN = nn.BatchNorm2d(3)
        # self.conv = torch.nn.Conv2d(1, 1, 1)
        # self.conv = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
        kernel_size=7
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)).double()
        # print(self.conv.weight.dtype)
        # print(self.conv.weight)
        # self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)).double()
        # print(self.conv.weight.dtype)
        # print(self.conv.weight)
        # self.conv.weight =  torch.nn.Parameter(torch.tensor([np.random.rand(1, kernel_size, kernel_size).astype(np.float32)]).double())
        # mid_val1 = np.random.rand(1,5)[0].astype(np.float32)
        # mid_val2 = np.random.rand(1,5)[0].astype(np.float32)
        # mid_val3 = np.random.rand(1,5)[0].astype(np.float32)
        # mid_val4 = np.random.rand(1, 5)[0].astype(np.float32)
        # mid_val5 = np.random.rand(1, 5)[0].astype(np.float32)
        # self.conv.weight =  torch.nn.Parameter(torch.tensor([[[mid_val1,
        #                                                        mid_val2,
        #                                                         mid_val3,
        #                                                         mid_val4,
        #                                                         mid_val5]]]).double())
                                    
        # self.conv.weight = torch.nn.Parameter(torch.tensor([[[[ 0.03307433053851127625, -0.13484150171279907227, -0.21625524759292602539], 
        #                                                         [ 0.14247404038906097412, -0.14247404038906097412, -0.24932956695556640625], 
        #                                                         [ 0.32311078906059265137, -0.14501821994781494141, -0.21371106803417205811]]]]))
        # self.conv.weight = torch.nn.Parameter(torch.tensor([[[[ 1.0, 1.0, -1.0], 
        #                                                         [ -1.0, 1.0, 1.0], 
        #                                                         [ 1.0, 1.0, -1.0]]]]))
        # self.conv.weight = torch.nn.Parameter(torch.tensor([[[[ 1.0, 1.0, -1.0], 
        #                                                          [ -1.0, 1.0, 1.0], 
        #                                                          [ 1.0, 1.0, 0.0]]]]).double())
        
        # self.conv.weight[0, 0, :, :] = torch.nn.Parameter(torch.tensor([[[[ -1.0, 0.0, 1.0], 
        #                                                         [ -1.0, 0.0, 1.0], 
        #                                                         [ -1.0, 0.0, 1.0]]]]).double(), requires_grad = True)
        self.conv.bias = torch.nn.Parameter(torch.zeros(8, ).double())
        # self.conv.bias = torch.nn.Parameter(torch.tensor(np.random.rand(1).astype(np.float32)).double())
        # self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point

    def forward(self, x):
        x = self.conv(x)
         
        return x



model_double = M2_double()
model_double.eval()

# print('Weight type ', model_double.conv.weight.dtype, "Bias type ", model_double.conv.weight.dtype)

# print("Weight ", model_double.conv.weight.detach())
print("Bias ", model_double.conv.bias.detach())

for i in range(100):
    input_fp32 = torch.rand(1, 3, 100, 100).double()

    # print("Input ", input_fp32)
    # input_fp32 = torch.tensor(np.random.rand(1,1,5,5))
    model_output_double = model_double(input_fp32).detach()
    convolved_img_torch_double = convolve_torch(input_fp32[0].detach(), model_double.conv.weight.detach(), model_double.conv.bias.detach())
    # convolved_img_torch_double = convolve_torch(input_fp32[0].detach(), model_double.conv.weight.detach(), model_double.conv.bias.detach())
    difference_torch_double = model_output_double[0] - convolved_img_torch_double

    # print(model_output_double[0,0,:])
    # print(difference_torch_double.shape)
    # print(difference_torch_double)
    print("Iter ", i, " Torch double model difference :", torch.sum(difference_torch_double))
