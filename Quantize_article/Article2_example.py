import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F
from utils import quantize_vector

from PIL import Image
import torchvision

# def quantize_vector(input_tensor, scale, zero_point):

#     integer_tensor = torch.clamp(torch.round(input_tensor.detach()/scale)+zero_point, min=0)

#     floating_point_tensor = (integer_tensor - zero_point)*scale.float()

#     return integer_tensor, floating_point_tensor


class Model_Quant_Dequant(nn.Module):

    def __init__(self):

        super(Model_Quant_Dequant, self).__init__()

        self.quant = QuantStub()
        self.fc7128 = nn.Sequential(nn.Linear(2048, 128))
        # self.conv.weight = torch.nn.Parameter(torch.tensor([[[[ 0.03307433053851127625, -0.13484150171279907227, -0.21625524759292602539], 
        #                                                             [ 0.14247404038906097412, -0.14247404038906097412, -0.24932956695556640625], 
        #                                                             [ 0.32311078906059265137, -0.14501821994781494141, -0.21371106803417205811]]]]))
        #     # self.conv.weight = torch.nn.Parameter(torch.tensor([[[[ 0.0, 0.0, 0.0], 
        #     #                                                         [ 0.0, 0.5, 0.0], 
        #     #                                                         [ 0.0, 0.0, 0.0]]]]))
        # self.conv.bias = torch.nn.Parameter(torch.tensor([0.0]))
        self.dequant = DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = torch.flatten(x, 1)
        x = self.dequant(x)

        return x

trfrm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.ToPILImage(mode="RGB"),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((160,160)),
        torchvision.transforms.ToTensor(),
        ])

input_fp32 = trfrm(Image.open('/home/avishek/FaceNet_Quant/datasets/Merged-short/7030/151105.jpg'))
input_fp32 = input_fp32.unsqueeze(0).to('cpu')

model_quant = Model_Quant_Dequant()
model_quant.eval()

model_quant.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_quant, inplace=True)
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.convert(model_quant, inplace=True)

output_quant_model = model_quant(input_fp32)

input_quant_manual_after_quant, input_quant_manual_after_quant_dequant = quantize_vector(input_fp32, model_quant.quant.scale, model_quant.quant.zero_point)

print("Final output comparison :", torch.sum(output_quant_model - input_quant_manual_after_quant_dequant))