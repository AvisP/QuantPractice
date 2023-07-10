import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision import datasets
# import torchvision.transforms as transforms
import os
import json
# import time
# import sys
import torch.quantization
import argparse
from loss import TripletLoss
from torch.nn.modules.distance import PairwiseDistance
# from data_loader import get_dataloader, get_dataloader_eval
from UpdatedDataloader import get_dataloader, get_dataloader_eval
from eval_metrics_modified import evaluate
from sklearn.utils import shuffle
import random

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

from torch.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F
from loss import TripletLoss

class NN1_BN_FaceNet_2K_160(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(NN1_BN_FaceNet_2K_160, self).__init__()
        # TODO: add BN layer before the first conv
        # remove all normalization/standardization of data
        # train with device 2
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
        #self.fc2 = nn.Sequential(nn.Linear(32*128, 32*128), nn.ReLU(inplace=True), nn.Dropout())
        self.fc7128 = nn.Sequential(nn.Linear(16*128, embedding_size))

        # D19 MODIFIED
        # self.fc1 = nn.Sequential(nn.Linear(256*7*7, 1*1024), nn.ReLU(inplace=True), nn.Dropout())
        # self.fc2 = nn.Sequential(nn.Linear(1*1024, 1*embedding_size), nn.ReLU(inplace=True), nn.Dropout())

        # VGG16 MODIFIED
        # self.fc1 = nn.Linear(256*7*7, 1*embedding_size), nn.ReLU(inplace=True)
        
        #self.softmax = nn.Softmax(dim=-1)

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
        # x = self.rnorm1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        # x = self.rnorm2(x)
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
        #x = torch.flatten(x, 1).unsqueeze(dim=2).unsqueeze(dim=3)

        x = self.fc1(x)

        #x = torch.flatten(x, 1)

        #x = self.fc2(x)

        x = self.fc7128(x)
        # x = self.softmax(x)
        
        x = self.dequant(x)
        x = nn.functional.normalize(x, p=2, dim=1)                
        return x
    
    def forward_classifier(self, x):
        features = self.forward(x)
        #res = self.model.classifier(features)
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

def eval_valid(model, triploss, dataloaders, data_size, phase=['valid']):
    for phase in phase:

        labels, distances = [], []
        triplet_loss_sum = 0.0
        l2_dist = PairwiseDistance(2)
        margin = 0.5 # args.margin   ## Include this as function input

        model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            if batch_idx%10 == 0:
                print('.', end = '')

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            # pos_cls = batch_sample['pos_class'].to(device)
            # neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # print(anc_embed)
                # print('After Norm')
                # print(nn.functional.normalize(torch.FloatTensor(anc_embed), p=2, dim=0))

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                # pos_hard_cls = pos_cls[hard_triplets]
                # neg_hard_cls = neg_cls[hard_triplets]

                model.forward_classifier(anc_hard_img)
                model.forward_classifier(pos_hard_img)
                model.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        _, _, accuracy, best_threshold, _, _, _ = evaluate(distances, labels, accuracy_only=True)
        # _, _, accuracy, _, _, _ = evaluate(distances, labels, accuracy)
        print('Inference - Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
        print('Inference - Accuracy           = {:.8f} %'.format(np.mean(accuracy)*100))
    return distances, labels, accuracy, best_threshold


def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):  
    labels, distances = [], []
    triplet_loss_sum = 0.0
    l2_dist = PairwiseDistance(2)
    margin = 0.5 # args.margin   ## Include this as function input

    l2_dist = PairwiseDistance(2)
    for phase in ['valid']:
        anc_sum = 0
        pos_sum = 0
        neg_sum = 0
        grad_sum = torch.tensor(0)
        labels, distances = [], []
        triplet_loss_sum = 0.0
        grad_sum.to(device)
        # if epoch != 0:
        #     previous_state_dict = model.module.state_dict()
        if phase == 'train':
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)
                
            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                anc_sum = anc_sum + anc_embed.sum()
                pos_sum = pos_sum + pos_embed.sum()
                neg_sum = neg_sum + neg_embed.sum()

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)
                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                model.forward_classifier(anc_hard_img)
                model.forward_classifier(pos_hard_img)
                model.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                # if phase == 'train':
                #     optimizer.zero_grad()
                #     triplet_loss.backward()
                #     optimizer.step()
                #     scheduler.step() # needs to be removed
                #     if scheduler.last_epoch % scheduler.step_size == 0:
                #         print("LR decayed to:", ', '.join(map(str, scheduler.get_last_lr())))

                # for params in model.parameters():
                #     s = abs(params.grad).sum()
                #     grad_sum = grad_sum + s
                # Sum of weights calculation and visualization 
                # start_filter = 0
                # current_state_dict = model.module.state_dict()
                # for i in range(start_filter,start_filter+16):
                #     current_state_dict['conv1.1.weight'][i] = previous_state_dict['conv1.1.weight'][i]
                #     current_state_dict['conv1.1.bias'][i] = previous_state_dict['conv1.1.bias'][i]

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        # tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        _, _, accuracy, best_threshold, _, _, _ = evaluate(distances, labels, accuracy_only=True)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))
    return distances, labels, accuracy, best_threshold


def load_model(saved_model_dir, checkpoint = './log/best_state.pth'):
    model = NN1_BN_FaceNet_2K_160(embedding_size=128).to(device)
    # checkpoint = './log/best_state.pth'
    checkpoint = saved_model_dir+'Facenet_2K_160.pth'
    print('loading', checkpoint)
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--config_file_location', default='/home/avishek/FaceNet_Quant/config.json', type=str,
                    help='configuration file location')

arg = parser.parse_args()

# parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

# parser.add_argument('--num_epochs', default=2000, type=int, metavar='NE',
#                     help='number of epochs to train (default: 200)')
# parser.add_argument('--num-valid-triplets', default=10000, type=int, metavar='NVT',
#                     help='number of triplets for validation (default: 10000)')
# parser.add_argument('--num-train-triplets', default=10000, type=int, metavar='NTT',
#                     help='number of triplets for training (default: 10000)')
# parser.add_argument('--batch_size', default=128, type=int, metavar='BS',
#                     help='batch size (default: 128)')
# parser.add_argument('--num-workers', default=8, type=int, metavar='NW',
#                     help='number of workers (default: 8)')
# parser.add_argument('--learning_rate', default=0.01, type=float, metavar='LR',
#                     help='learning rate (default: 0.001)')
# parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
#                     help='margin (default: 0.5)')
# parser.add_argument('--train-root-dir', default='/dataset/VGG_Face2/train_images_aligned/', type=str,
#                     help='path to train root dir')
# parser.add_argument('--train-csv-name', default='/home/avishek/facenet-final/datasets/merged.csv', type=str,
#                     help='list of training images')
# parser.add_argument('--valid-root-dir', default='/dataset/VGG_Face2/test_images_aligned/', type=str,
#                     help='path to valid root dir')
# parser.add_argument('--valid-csv-name', default='/home/avishek/facenet-final/datasets/lfw-dataset.csv', type=str, 
#                     help='list of validation images')
# parser.add_argument('--total_parts', default=3, type=int, 
#                     help='number of parts')
file_location = arg.config_file_location

f = open(file_location)
data = json.load(f)
f.close()

args = argparse.Namespace(**data)
# args, unknown = parser.parse_known_args()

data_path = ' /dataset/VGG_Face2/train_images_aligned/output'
saved_model_dir = '/home/avishek/FaceNet_Quant/saved_model/'
float_model_dir = '/home/avishek/FaceNet_Quant/float_model_weight/'
float_model_file = 'Facenet_2K_160.pth'
# scripted_float_model_file = 'facenet_quantization_scripted.pth'
# scripted_quantized_model_file_S1 = 'faceenet_quantized_S1_05K_perchannel.pth'
scripted_quantized_model_file_S2 = 'Facenet_2K_160_perchannel.pth'

# print('Batch size: ', args.batch_size)
float_model = load_model(float_model_dir).to('cpu')
float_model.eval()
print('Model Loading complete')
# float_model.conv1[0] = nn.Identity()
print(float_model.conv1)

# Fuse Conv, bn and relu
float_model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
float_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(float_model.qconfig)
torch.quantization.prepare(float_model, inplace=True)
print('Model Fusion and configuration setting complete')

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Model:After observer insertion \n\n', float_model.conv1)

triplet_loss = TripletLoss(args.margin).to(device)
start_epoch = 0

parts = args.total_parts

# Calibrate with the training set
print('Starting Post Training Quantization Calibration')

for epoch in range(start_epoch, args.num_epochs):
    print(f"{epoch}th span over the entire data")

    if args.loadlast:
        if start_epoch != epoch:
            start_part = 1
    # wandb_total_epoch = args.wandb_total_epoch
        # start_mini_epoch = checkpoint['epoch'] + 1
    start_part = (args.wandb_total_epoch // args.n)

    if start_part > args.total_parts:
        start_part = (start_part % args.total_parts) + 1
    else:
        start_part += 1

    for part in range(start_part,parts+1):
        if args.loadlast:
            if start_part != part:
                start_mini_epoch = 0

        print(f'Loading data for part : {part}')

        array_of_images = np.load(args.train_file_location +'/array_of_images_'+str(part)+'.npy', mmap_mode = 'r+',fix_imports=False)
        pos_img = np.load(args.train_file_location +'/Triplets/pos_img_'+str(part)+'.npy', mmap_mode = 'r+',fix_imports=False)
        neg_img = np.load(args.train_file_location +'/Triplets/neg_img_'+str(part)+'.npy',mmap_mode = 'r+',fix_imports=False)
        anc_img = np.load(args.train_file_location +'/Triplets/anc_img_'+str(part)+'.npy',mmap_mode = 'r+',fix_imports=False)
        pos_img, neg_img, anc_img = shuffle(pos_img,neg_img,anc_img)
        start_mini_epoch = 0;

        for mini_epoch in range(start_mini_epoch,len(array_of_images)//args.num_train_triplets):
            print(80 * '=')
            print('Mini Epoch       = {:d}'.format(mini_epoch))
            valid_epoch = random.randint(0,(len(array_of_images)//args.num_valid_triplets)-1)
           
            data_loaders, data_size = get_dataloader(args.num_train_triplets, args.num_valid_triplets,
                                                    args.batch_size, args.num_workers, array_of_images = array_of_images,
                                                    pos_img = pos_img, neg_img = neg_img, anc_img = anc_img, epoch= mini_epoch,
                                                    valid_epoch=valid_epoch, 
                                                    test_images=array_of_images, test_pos_img =pos_img, test_neg_img=neg_img, test_anc_img=anc_img)

            _, _, accuracy_train, best_threshold_train  = train_valid(float_model, 0, triplet_loss, 0, mini_epoch, data_loaders, data_size)

# print('Starting DataLoading')
# data_loaders_train, data_size_train = get_dataloader(args.train_root_dir, args.valid_root_dir,
#                                                  args.train_csv_name, args.valid_csv_name,
#                                                  args.num_train_triplets, args.num_valid_triplets,
#                                                  args.batch_size, args.num_workers)

# data_loaders_test, data_size_test = get_dataloader_eval(args.valid_root_dir,
#                                                 args.valid_csv_name,
#                                                 args.num_valid_triplets,
#                                                 args.batch_size, args.num_workers)
# print('Finished DataLoading')


# _, _, accuracy_train, best_threshold_train = eval_valid(float_model, triplet_loss, data_loaders_train, data_size_train, phase=['train'])
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.backends.quantized.engine = 'qnnpack'
torch.quantization.convert(float_model, inplace=True)
print('Post Training Quantization: Convert done')
print('\n After fusion and quantization, note fused modules: \n\n',float_model.conv1)

print("Size of model after quantization")
print_size_of_model(float_model)

print("Loading data for inference ......")
test_images = np.load(args.test_file_location +'/array_of_images_1.npy', mmap_mode = 'r+',fix_imports=False)
test_pos_img = np.load(args.test_file_location +'/Triplets/pos_img_1.npy',mmap_mode = 'r+',fix_imports=False)
test_anc_img = np.load(args.test_file_location +'/Triplets/anc_img_1.npy',mmap_mode = 'r+',fix_imports=False)
test_neg_img = np.load(args.test_file_location +'/Triplets/neg_img_1.npy',mmap_mode = 'r+',fix_imports=False)

data_loaders_test, data_size_test = get_dataloader_eval(len(test_images),
                args.batch_size, args.num_workers, test_images,
                test_pos_img, test_neg_img, test_anc_img,
                valid_epoch=0)

# Accuracy of quantized model on validation Set
print('Evaluating calibrated static quantized model on test set')
_, _, accuracy_test, best_threshold_test = eval_valid(float_model, triplet_loss, data_loaders_test, data_size_test)

# Accuracy_Threshold_Table = pd.DataFrame(np.array(np.transpose(np.array([accuracy_train, best_threshold_train, accuracy_test, best_threshold_test]))), 
#                 columns=[ 'Train_Accuracy', 'Train_Threshold', 'Test_Accuracy', 'Test_Threshold'])
# Accuracy_Threshold_Table.to_csv(saved_model_dir+'Acc_Thres_Train_PT_'+str(np.round(np.mean(accuracy_train), 3))+'_Test_'+str(np.round(np.mean(accuracy_test), 3))+'.csv')
torch.save({'state_dict': float_model.state_dict()}, (saved_model_dir + scripted_quantized_model_file_S2[:-4]+'_PT_Train_'+str(np.round(np.mean(accuracy_train), 3))+'_Test_'+str(np.round(np.mean(accuracy_test), 3))+scripted_quantized_model_file_S2[-4:]))
#torch.save({'state_dict': float_model.state_dict()}, (saved_model_dir + scripted_quantized_model_file_S2[:-4]+'_PT_Train_'+str(np.round(np.mean(accuracy_train), 3))+'_Test_'+str(np.round(np.mean(accuracy_test), 3))+scripted_quantized_model_file_S2[-4:]))
# torch.save(float_model.state_dict(), saved_model_dir + scripted_quantized_model_file_S2[:-4]+'_Train_'+str(np.round(np.mean(accuracy_train), 3))+'_Test_'+str(np.round(np.mean(accuracy_test), 3))+'_M2'+scripted_quantized_model_file_S2[-4:])