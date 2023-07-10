import torch
import numpy as np
import sys

def convolve_torch(input_image: torch.tensor, kernel: torch.tensor, bias: torch.tensor, stride=1) -> torch.tensor:    
    ### Padding for kernel size (1 ,3, 5, 7)  should be (0, 2*1, 2*2, 2*3)
    ### Add description of what dimension is what
    k = kernel.shape[2]
    number_input_row = input_image.shape[1]
    number_input_col = input_image.shape[2]
    input_depth = input_image.shape[0]   # R, G, B channel
    filter_depth = kernel.shape[1]       # filter depth
    filter_num = kernel.shape[0]         # number of filters
    bias_depth = bias.shape[0]

    if input_depth != filter_depth:
        print("Error: Number of channels in both image and filter depth must match.")
        sys.exit()

    if bias_depth != filter_num:
        print("Error: Bias depth and filter number must match.")
        sys.exit()
    
    if torch.isnan(input_image.all()):
        print("Input image has NaN values")
        sys.exit()

    if torch.isnan(kernel.all()):
        print("Kernel has NaN values")
        sys.exit()

    # print("Input Image dtype", input_image.dtype, "Kernel dtype", kernel.dtype, "Bias dtype", bias.dtype)

    if input_image.dtype != kernel.dtype:
        print(" Warning: Input Image and Kernel data types don't match")
        print("Input Image dtype", input_image.dtype, "Kernel dtype", kernel.dtype)

    if input_image.dtype != bias.dtype:
        print(" Warning: Input Image and Bias data types don't match")
        print("Input Image dtype", input_image.dtype, "Bias dtype", bias.dtype)
    
    if str(input_image.dtype) == 'torch.float32':
        # print("Inside torch float32")
        padded_image = torch.zeros((input_depth, number_input_row+(k-1), number_input_col+(k-1)))
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = torch.zeros((filter_num, int(number_input_row/stride), int(number_input_col/stride)))
    elif str(input_image.dtype) == 'torch.float64':
        # print("Inside torch float64")
        padded_image = torch.zeros((input_depth, number_input_row+(k-1), number_input_col+(k-1))).double()
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = torch.zeros((filter_num, int(number_input_row/stride), int(number_input_col/stride))).double()
    elif str(input_image.dtype) == 'torch.float16':
        padded_image = torch.zeros((input_depth, number_input_row+(k-1), number_input_col+(k-1))).half()
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = torch.zeros((filter_num, int(number_input_row/stride), int(number_input_col/stride))).half()
    else :
        print("Error: Input image is not of float, double or half tensor")
        sys.exit()
    
    # val_matrix = torch.zeros(filter_depth, k, k)
    
    for f in range(filter_num):
        convolved_img[f,:,:] = bias[f]
        i_idx = 0
        for i in range(0, number_input_row, stride):  # Increment i by stride  - 1
            j_idx = 0
            for j in range(0, number_input_col, stride):   # Increment j by stride  - 1
                mat = padded_image[:, i:i+k, j:j+k]
                # print(mat.shape, kernel[f,:,:,:].shape)
                # val = torch.sum(torch.multiply(mat, kernel[f,:,:,:]))
                
                val_matrix = torch.multiply( mat, kernel[f,:,:,:])
                # if i_idx==4 and j==3:
                    # print(mat.shape, kernel[f,:,:,:].shape)
                #     # print(i_idx, j_idx)
                #     print("Val matrix", val_matrix.shape, val_matrix)
                #     print("Torch sum", torch.sum(val_matrix)+bias[f])
                #     sum_val = 0
                #     for row in range(k):
                #         for column in range(k):
                #             sum_val = (sum_val + val_matrix[0, row, column]).double()
                #     print("Sum val ", sum_val)
                # print( kernel[f,:,:,:].dtype, mat.dtype)
                # sum_val = 0
                # for depth in range(filter_depth):
                #     for row in range(k):
                #         for column in range(k):
                #             sum_val = sum_val + val_matrix[depth, row, column]
                # print("i= ",i, "j= ", j, "i_idx = ", i_idx, "j_idx = ", j_idx, "val = ", val, "mat = ", mat)
                convolved_img[f, i_idx, j_idx ] = convolved_img[f, i_idx, j_idx ] + torch.sum(val_matrix)
                # convolved_img[f, i_idx, j_idx ] = torch.sum(val_matrix)#sum_val #np.sum(np.multiply(mat, kernel[:,:,:,f]))
                # print(i_idx, j_idx, sum_val)
                j_idx += 1
            i_idx += 1
        
        # convolved_img[f, :, :] = convolved_img[f, :, :] + bias[f]

    if torch.isnan(convolved_img.all()):
        print("Convolved image has NaN values")
        sys.exit()
    # print(convolved_img.dtype, padded_image.dtype)
    return convolved_img



def convolve_numpy(input_image: np.array, kernel: np.array, bias: np.array, stride=1) -> np.array:    
    ### Padding for kernel size (1 ,3, 5, 7)  should be (0, 1 , 2, 3)
    ### Add description of what dimension is what
    k = kernel.shape[2]
    number_input_row = input_image.shape[1]
    number_input_col = input_image.shape[2]
    input_depth = input_image.shape[0]   # R, G, B channel
    filter_depth = kernel.shape[1]       # filter depth
    filter_num = kernel.shape[0]         # number of filters
    bias_depth = bias.shape[0]

    if input_depth != filter_depth:
        print("Error: Number of channels in both image and filter depth must match.")
        sys.exit()

    if bias_depth != filter_num:
        print("Error: Bias depth and filter number must match.")
        sys.exit()
    
    if np.isnan(input_image.all()):
        print("Input image has NaN values")
        sys.exit()

    if np.isnan(kernel.all()):
        print("Kernel has NaN values")
        sys.exit()

    if input_image.dtype != kernel.dtype:
        print(" Warning: Input Image and Kernel data types don't match")

    if input_image.dtype != bias.dtype:
        print(" Warning: Input Image and Bias data types don't match")

    if str(input_image.dtype) == 'float64':
        # print('Inside float64')
        padded_image = np.zeros(shape=(input_depth, number_input_row+int((k+1)/2), number_input_col+int((k+1)/2))).astype(np.float64)
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = np.zeros(shape=(filter_num, int(number_input_row/stride), int(number_input_col/stride))).astype(np.float64)
    elif str(input_image.dtype) == 'float32':
        # print('Inside float32')
        padded_image = np.zeros(shape=(input_depth, number_input_row+int((k+1)/2), number_input_col+int((k+1)/2))).astype(np.float32)
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = np.zeros(shape=(filter_num, int(number_input_row/stride), int(number_input_col/stride))).astype(np.float32)
    elif str(input_image.dtype) == 'float16':
        padded_image = np.zeros(shape=(input_depth, number_input_row+int((k+1)/2), number_input_col+int((k+1)/2))).astype(np.float16)
        padded_image[:, ((k-1)>>1):number_input_row + ((k-1)>>1), ((k-1)>>1):number_input_col + ((k-1)>>1)] = input_image
        convolved_img = np.zeros(shape=(filter_num, int(number_input_row/stride), int(number_input_col/stride))).astype(np.float16)
    else :
        print("Error: Input image is not of float, double or half")
        sys.exit()

    # print("Input Image dtype", input_image.dtype)
    # print("Kernel dtype", kernel.dtype)
    # print("Bias dtype", bias.dtype)

    # print("Padded image type", padded_image.dtype)
    # print("inside function con image type", convolved_img.dtype)
    # for f in range(filter_num):
    #     for i in range(number_input_row):  # Increment i by stride  - 1
    #         for j in range(number_input_col):   # Increment j by stride  - 1
    #             mat = padded_image[i:i+k, j:j+k, :]
    #             convolved_img[i, j, f ] = np.sum(np.multiply(mat, kernel[:,:,:,f]))
    # print(padded_image.shape)
    for f in range(filter_num):
        # print("f = ", f, kernel[f,:,:,:])
        i_idx = 0
        for i in range(0, number_input_row, stride):  # Increment i by stride  - 1
            j_idx = 0
            for j in range(0, number_input_col, stride):   # Increment j by stride  - 1
                mat = padded_image[:, i:i+k, j:j+k]
                val = np.sum(np.multiply(mat, kernel[f,:,:,:]))
                # print("i= ",i, "j= ", j, "i_idx = ", i_idx, "j_idx = ", j_idx, "val = ", val, "mat = ", mat)
                convolved_img[f, i_idx, j_idx ] = val#np.sum(np.multiply(mat, kernel[:,:,:,f]))
                j_idx += 1
            i_idx += 1
        
        convolved_img[f, :, :] += bias[f]

    # print("inside function mat image type", mat.dtype)
    # print("inside function val image type", val.dtype)
    # print("inside function convolved image type", convolved_img.dtype)

    if np.isnan(convolved_img.all()):
        print("Convolved image has NaN values")
        sys.exit()

    return convolved_img