# QuantPractice
This repo contains some qunatization scripts and notebooks of a custom Facenet model and a custom python written quantized convolution block that doesn't use pytorch blocks. Results have been validated against the pytorch model. The weights and data paths on which training is done is nto provided

## Quant_Initial
This folder contains some some scripts and notebooks that replicates the MobileNet quantization process described on pytorch website using Static Quantization 
https://pytorch.org/blog/quantization-in-practice/

Also it uses the techniques in the examples and applies them to the custom facenet model
-  Quant_facenet_S2_2K_160.py - Post Training Dynamic Quantization
-  Quant_facenet_S3_2K_160.py - Quantization Aware Training

## ManualQuantFolder

This folder contains manual code to replicate the Quantized Pytorch model using pythn code only. Purpose of doing this is to export the model to C format later on for deploying on edge devices.

- FaceNet_fullmodel_detail.py, FaceNet_fullmodel_detail.py - These are the final scripts for validating the custom model with the quantized Pytorch model
- Float Model validate 2.py - Scripts for validating the custom model with floating point model
- QuantLayer_FloatingPoint_3.ipynb, QuantLayer_FullWeight_Quantized 2.ipynb, QuantLayerCheck.ipynb - Some additional notebooks that are used for testing out the BatchNorm block, floating point and quantization model
- custom_convolve.py - contains the custom written convolution block

## Quantize_article
* Work in Progreess*
This folder contains small code sections that demonstrates quantization process of Convolution block in ease of difficulty from small weight block with single non-zero element to full weight matrix and then combining convolve blocks with Batch Norm layers. Purpose is to share some some tips and tricks associated with quantization process.
