from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import sys,os
from facenet_pytorch.models.mtcnn import MTCNN, fixed_image_standardization
from time import time
import glob
from PIL import Image, ImageDraw

workers = 0 if os.name == 'nt' else 4

trans = transforms.Compose([
    transforms.Resize(512)
])

trans_cropped = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder('/dataset/VGG_Face2/data/train', transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}

def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn_pt = MTCNN(device=torch.device(device))

save_path_loc = '/home/avishek/Quantization/data/train_images_aligned/'

names = []
aligned = []
aligned_fromfile = []
cur_idx = 0
file_counter = -1

for img, idx in dataset:
    file_counter += 1
    if idx>1802:        
        name = dataset.idx_to_class[idx]
        sample_folder, _ = dataset.samples[file_counter]
        sample_fname = sample_folder.split('/')[-1].split('_')[0]
        print('Processing ', sample_folder)
        if int(sample_fname) > 0 and not os.path.exists(save_path_loc+name+'/'+sample_fname+'.png'):            
            start = time()
            img_align = mtcnn_pt(img, save_path=save_path_loc+name+'/'+sample_fname+'.png')
            print('MTCNN time: {:6f} seconds'.format(time() - start))
            print('Cropping and saving at', save_path_loc+name+'/'+sample_fname)
            # Comparison between types
            img_box = mtcnn_pt.detect(img)[0]
            # if type(img_box) is np.ndarray:
                # assert (img_box - mtcnn_pt.detect(np.array(img))[0]).sum() < 1e-2
                # assert (img_box - mtcnn_pt.detect(torch.as_tensor(np.array(img)))[0]).sum() < 1e-2

                # # Batching test
                # assert (img_box - mtcnn_pt.detect([img, img])[0]).sum() < 1e-2
                # assert (img_box - mtcnn_pt.detect(np.array([np.array(img), np.array(img)]))[0]).sum() < 1e-2
                # assert (img_box - mtcnn_pt.detect(torch.as_tensor([np.array(img), np.array(img)]))[0]).sum() < 1e-2
            # else:
            #     print('Image box not found for', name+'/'+sample_fname)
                # # Box selection
                # mtcnn_pt.selection_method = 'probability'
                # print('\nprobability - ', mtcnn_pt.detect(img))
                # mtcnn_pt.selection_method = 'largest'
                # print('largest - ', mtcnn_pt.detect(img))
                # mtcnn_pt.selection_method = 'largest_over_theshold'
                # print('largest_over_theshold - ', mtcnn_pt.detect(img))
                # mtcnn_pt.selection_method = 'center_weighted_size'
                # print('center_weighted_size - ', mtcnn_pt.detect(img))

            # if img_align is not None:
            #     names.append(name)
            #     aligned.append(img_align)
            #     aligned_fromfile.append(get_image(save_path_loc+name+'/'+sample_fname+'.png', trans_cropped))
        else:
            print('Skipping this file')
