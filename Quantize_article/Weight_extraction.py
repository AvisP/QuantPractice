import torch

checkpoint = torch.load('/home/avishek/Quantization/model_weights/Facenet_2K_160_perchannel_PT_Train_0.854_Test_0.844.pth', map_location=torch.device('cpu'))
#model.load_state_dict(checkpoint['state_dict'])

#checkpoint['state_dict']

from collections import OrderedDict

test_dict = OrderedDict()

for key, value in checkpoint['state_dict'].items():
    # if key.__contains__('conv1') or key.__contains__('quant') or key.__contains__('conv2'):
    if key.__contains__('fc7128'):
        test_dict[key] = value
        print(key)


torch.save({'state_dict': test_dict},"last_layer.pt")