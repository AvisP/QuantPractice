import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torchvision
# from torch.utils.data import DataLoader
# from torchvision import datasets
import torchvision.transforms as transforms
import os
# import time
# import sys
import torch.quantization
import argparse
# from loss import TripletLoss
import matplotlib.pyplot as plt
from PIL import Image
from facenet import NN1_BN_FaceNet_4K_M_Quantized, NN1_BN_FaceNet_2K_Quantized, NN1_BN_FaceNet_05K_Quantized
from facenet import NN1_BN_FaceNet_4K_M, NN1_BN_FaceNet_2K

# verified_image_path2 = os.path.abspath('./datasets/lfw-short/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')  ## Change here
bright_image_path = os.path.abspath('/dataset/Merged/4449941/027.jpg')  ## Change here
# verified_image_path2 = os.path.abspath('/dataset/Merged/0863599/001.jpg')
dark_image_path = os.path.abspath('/dataset/Merged/4449941/015.jpg')

# verified_image_path2 = os.path.abspath('./datasets/Beyza_cropped.jpg')  ## Change here
# verified_image_path = os.path.abspath('./datasets/Devyan_cropped.jpg')  ## Change here

trfrm = torchvision.transforms.Compose([
            #torchvision.transforms.ToTensor(),
            #torchvision.transforms.ToPILImage(mode="RGB"),
            torchvision.transforms.Resize((220,220)),
            torchvision.transforms.ToTensor(),
            ])

# model = NN1_BN_FaceNet_05K_Quantized(embedding_size=128)
# model = NN1_BN_FaceNet_4K_M_Quantized(embedding_size=128)
model = NN1_BN_FaceNet_2K_Quantized(embedding_size=128)
# model = NN1_BN_FaceNet_4K_M(embedding_size=128)

model.to('cpu')
threshold = 0.835
#model = torch.nn.DataParallel(model)
model.eval()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
torch.backends.quantized.engine = 'qnnpack'
# torch.backends.quantized.engine = 'fbgemm'    
torch.quantization.convert(model, inplace=True)
print('Model Fusion and configuration setting complete')

# checkpoint_path = '/home/avishek/facenet-final/float_model_weight/NN1_BN_FaceNet_4K_M.pth'
# checkpoint_path = './saved_model/faceenet_quantized_S2_05K_Train_0.829_Test_0.835.pth'    ## Change here
# checkpoint_path = '/home/avishek/facenet-final/saved_model/faceenet_quantized_S2_4K_Train_0.887_Test_0.82.pth'
# checkpoint_path = '/home/avishek/facenet-final/saved_model/faceenet_quantized_S2_4K_Devyan_Train_0.886_Test_0.818.pth'
# checkpoint_path = '/home/avishek/facenet-final/saved_model/faceenet_quantized_S2_4K_Devy_Train_0.891_Test_0.814.pth'

checkpoint_path = '/home/avishek/FaceNet_Quant/saved_model/faceenet_quantized_S2_2K_perchannel_PT_Train_0.731_Test_0.896.pth'#faceenet_quantized_S2_4K_perchannel_PT_Train_0.887_Test_0.821.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))   
model.load_state_dict(checkpoint['state_dict'])
# torch.jit.save(torch.jit.script(model), 'jist_saved_model.pth')

# bright_image = trfrm(Image.open(bright_image_path))
# bright_image = bright_image.unsqueeze(0).to('cpu')
# # print(image.shape)
# dark_image = trfrm(Image.open(dark_image_path))
# dark_image = dark_image.unsqueeze(0).to('cpu')
# print(model(image2))
# plt.imshow(Image.open(verified_image_path2))
# imgplot = plt.imshow(image2.reshape(220,220))
# plt.show()
# print(torch.nn.functional.pairwise_distance(model(image2), model(image)))

train_file_location = "/dataset/Merged/10000000/"

filelist = []
for (dirpath, dirnames, filenames) in os.walk(train_file_location):
    filelist.extend(filenames)

output_vector_quant_model = []
for file in filelist:
    current_image = trfrm(Image.open(train_file_location+file))
    current_image = current_image.unsqueeze(0).to('cpu')
    output_vector_quant_model.append(model(current_image))

dist_mat_array_quant = []
for i in range(0, len(output_vector_quant_model)):
    for j in range(i+1, len(output_vector_quant_model)):
        # Dist_matrix[i][j] = np.linalg.norm(output_vector[i] - output_vector[j])
        dist_mat_array_quant.append(np.linalg.norm(output_vector_quant_model[i] - output_vector_quant_model[j]))

float_model = NN1_BN_FaceNet_2K(embedding_size=128)
float_model.to('cpu')
float_model.eval()
# checkpoint_path = './float_model_weight/NN1_BN_FaceNet_4K_M.pth'
checkpoint_path = './float_model_weight/NN1_BN_FaceNet_2K.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))   
float_model.load_state_dict(checkpoint['state_dict'])

output_vector_float_model = []
for file in filelist:
    current_image = trfrm(Image.open(train_file_location+file))
    current_image = current_image.unsqueeze(0).to('cpu')
    output_vector_float_model.append(float_model(current_image).detach().numpy())

dist_mat_array_float = []
for i in range(0, len(output_vector_float_model)):
    for j in range(i+1, len(output_vector_float_model)):
        dist_mat_array_float.append(np.linalg.norm(output_vector_float_model[i] - output_vector_float_model[j]))

negative_file_location = "/dataset/Merged/" #/home/avishek/MergedTest/"

negative_filelist = []
output_vector_negative_quant_model = []
output_vector_negative_float_model = []

dist_mat_negative_quant = np.empty((len(filelist),),dtype=object)
for i,v in enumerate(dist_mat_negative_quant):
    dist_mat_negative_quant[i]=[0]

dist_mat_negative_float = np.empty((len(filelist),),dtype=object)
for i,v in enumerate(dist_mat_negative_float):
    dist_mat_negative_float[i]=[0]

for r, d, f in os.walk(negative_file_location):
    for file in f:
        if file.endswith(".jpg") or file.endswith(".png"):
            if 'Devyan' not in file:
                print(os.path.join(r, file))
                negative_filelist.append(os.path.join(r, file))
                current_image = trfrm(Image.open(os.path.join(r, file)))
                current_image = current_image.unsqueeze(0).to('cpu')
                output_vector_negative_quant_model = model(current_image)
                output_vector_negative_float_model = float_model(current_image).detach().numpy()
                for k, indv_list in enumerate(dist_mat_negative_quant):
                    indv_list.append(np.linalg.norm(output_vector_quant_model[k] - output_vector_negative_quant_model))
                for k, indv_list in enumerate(dist_mat_negative_float):
                    indv_list.append(np.linalg.norm(output_vector_float_model[k] - output_vector_negative_float_model))
            else:
                print('Skipping as anchor image')

with open("dist_mat_negative_float", "wb") as fp:   #Pickling
    pickle.dump(dist_mat_negative_float, fp)
with open("dist_mat_negative_quant", "wb") as fp1:   #Pickling
    pickle.dump(dist_mat_negative_quant, fp1)

# with Image.open(train_file_location+file) as im:
#     im.save("abcd.JPEG")

# verified_embed_image = model(bright_image)
# print('Quantized model bright image')
# print(model(bright_image))
# print('Quantized model dark image')
# print(model(dark_image))


# float_model = NN1_BN_FaceNet_4K_M(embedding_size=128)
# float_model.to('cpu')
# float_model.eval()
# checkpoint_path = './float_model_weight/NN1_BN_FaceNet_4K_M.pth'
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))   
# float_model.load_state_dict(checkpoint['state_dict'])

# # float_image = trfrm(Image.open(verified_image_path2))
# # float_image = float_image.unsqueeze(0).to('cpu')
# print('Float model bright image')
# print(float_model(bright_image))
# print('Float model dark image')
# print(float_model(dark_image))

# print(torch.nn.functional.pairwise_distance(model(image2), float_model(float_image)))
