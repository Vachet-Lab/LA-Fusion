import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import glob
import re
import SimpleITK as sitk
import scipy.stats
import matplotlib as mpl
import pandas as pd
import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import colorspacious
import os
from scipy import ndimage
from sklearn import manifold
from IPython.display import display
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import linear_model
from skimage.metrics import structural_similarity as ssim
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from skimage.util import img_as_float
from matplotlib.colors import ListedColormap


def hotspot_removal_conditions(quantile, MALDI_rot):
    Quantile_99 = np.quantile(MALDI_rot, quantile) 

    MALDI_image_hot = MALDI_rot.copy()
    MALDI_image_hot[MALDI_image_hot > Quantile_99] = Quantile_99

    row_hot, col_hot = MALDI_image_hot.shape

    MALDI_vector_raw = MALDI_rot.reshape(row_hot*col_hot)
    MALDI_vector_hot = MALDI_image_hot.reshape(row_hot*col_hot)

    plt.figure(figsize=(18, 10))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(MALDI_rot)
    plt.axis('off')
    plt.title('Before hotspot removal')
    ax = plt.subplot(2, 2, 2)
    plt.boxplot(MALDI_vector_raw)
    plt.title('Before hotspot removal')
    ax = plt.subplot(2, 2, 3)
    plt.imshow(MALDI_image_hot)
    plt.axis('off')
    plt.title('After hotspot removal')
    ax = plt.subplot(2, 2, 4)
    plt.boxplot(MALDI_vector_hot)
    plt.title('After hotspot removal')
    plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text)]

def images_plot(images, image_columns):
         
    # Images of the selected signals
    length = len(images)
    rows_graph = math.ceil(length/image_columns)

    plt.figure(figsize=(18, 28))
    for n,im in enumerate(images):
        ax = plt.subplot(rows_graph, image_columns, (n+1))
        plt.imshow(im)
        plt.axis('off')
        plt.title('Image {0}'.format(n+1))
    plt.show()

def LA_process(foldername,extension):    
    files = glob.glob1(foldername,extension)
    files.sort(key=natural_keys)
    LA_images = []
    for file in files:
        LA_image = np.loadtxt(foldername + file, delimiter=',')
        LA_images.append(LA_image)
    return LA_images

def hotspot_removal(images, quantile, BM_LA_path):
    BM_LA = np.loadtxt(BM_LA_path, delimiter=',')
    BM_LA[BM_LA == 255] = 1
    images_hotspot = []
    for n,im in enumerate(images):
        Q_im = np.quantile(im, quantile) 
        im[im > Q_im] = Q_im
        im_BS = im * BM_LA
        images_hotspot.append(im_BS)
    return images_hotspot, BM_LA

def select_LA_channel(images_processed_LA, nmetals,min_value,max_value,plt_color):
    LA_image = images_processed_LA[nmetals-1]
    plt.imshow(LA_image, vmin=min_value,vmax=max_value, cmap=plt_color)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    dimension = LA_image.shape
    print("LA image size:", dimension)
    return LA_image

def populate_border(matrix):
    '''
        Function populate_border used to fine tune, delimitate border of the tissue sample, based on any metal content. This function is
        concatenated with the remove_background function
    
        Input:
            matrix = np array, correspond to the matrix index_threshold. This is the matrix that have the applied condition
            matrix < threshold, this matrix correspond to a boolean matrix which have defined True and False values.
    
        Output:
            border = np array 
    '''
    border = np.ones(matrix.shape)
    for n in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            if matrix[n, m] == True:
                border[n, m] = 0
            elif matrix[n, m] == False:
                break   
    for n in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            if matrix[n, matrix.shape[1] - m - 1] == True:
                border[n, matrix.shape[1] - m - 1] = 0
            elif matrix[n, matrix.shape[1] - m - 1] == False:
                break
    for m in range(matrix.shape[1]):
        for n in range(matrix.shape[0]):
            if matrix[n, m] == True:
                border[n, m] = 0
            elif matrix[n, m] == False:
                break
    for m in range(matrix.shape[1]):
        for n in range(matrix.shape[0]):
            if matrix[matrix.shape[0] - n - 1, m] == True:
                border[matrix.shape[0] - n - 1, m] = 0
            elif matrix[matrix.shape[0] - n - 1, m] == False:
                break
    return border

def populate_border(matrix):
    '''
        Function populate_border used to fine tune, delimitate border of the tissue sample, based on any metal content. This function is
        concatenated with the remove_background function
    
        Input:
            matrix = np array, correspond to the matrix index_threshold. This is the matrix that have the applied condition
            matrix < threshold, this matrix correspond to a boolean matrix which have defined True and False values.
    
        Output:
            border = np array 
    '''
    border = np.ones(matrix.shape)
    for n in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            if matrix[n, m] == True:
                border[n, m] = 0
            elif matrix[n, m] == False:
                break   
    for n in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            if matrix[n, matrix.shape[1] - m - 1] == True:
                border[n, matrix.shape[1] - m - 1] = 0
            elif matrix[n, matrix.shape[1] - m - 1] == False:
                break
    for m in range(matrix.shape[1]):
        for n in range(matrix.shape[0]):
            if matrix[n, m] == True:
                border[n, m] = 0
            elif matrix[n, m] == False:
                break
    for m in range(matrix.shape[1]):
        for n in range(matrix.shape[0]):
            if matrix[matrix.shape[0] - n - 1, m] == True:
                border[matrix.shape[0] - n - 1, m] = 0
            elif matrix[matrix.shape[0] - n - 1, m] == False:
                break
    return border

def remove_background(final_matrices,line,std_threshold):   
    '''
        Function remove_background used to calculate the average and std of the background and set the theshold values
    
        Input:
            matrix = np array, data matrix with the Zn data final_matrices[Zn_index]
            line = int, index of the line that will be used to perform the background calculation, usually 0 
            tolerance_std = int, tolerance of the std, usually is 3 

        Output:
            background_mask = np array, background mask of the image data 
    '''
    matrix = final_matrices
    average_col = np.mean(matrix[:, line-1])
    std_col = np.std(matrix[:, line-1])
    average_row = np.mean(matrix[line-1, :])        
    std_row = np.std(matrix[line-1, :])   
    if std_col < std_row:
        average = average_col
        std = std_col    
    else:
        average = average_row
        std = std_row 
    threshold = average + std_threshold*std    
    index_threshold = matrix < threshold
    background_mask = populate_border(index_threshold)
    return background_mask

def generate_background_plot(background_mask):   
    '''
        Function generate_background_plot used to generate a plot of the background mask

        Input:
            background_mask = np array, background mask of the image data
        
        Output:
            matplotlib inline image of the background mask
    '''
    fig = plt.figure(figsize=[5,4])    
    ax = fig.add_subplot(1,1,1)
    plt.imshow(background_mask, interpolation='None', cmap=plt.cm.hot)
    plt.title('Background mask')
    plt.axis('off')
    plt.show()
    
    
def convert_to_greyscale(image):
    gray_image = image.convert('L')
    plt.imshow(gray_image, vmin=0,vmax=255, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    np_gray_image = np.asarray(gray_image)
    dimension = np_gray_image.shape
    print("AF image size:", dimension)
    return gray_image 
    
    
def rezise_AF_image(x, y, gray_image):
    new_resolution = (x, y)
    resized_image = gray_image.resize(new_resolution)
    AF_image = np.asarray(resized_image)
    plt.imshow(AF_image, vmin=0,vmax=255, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.axis('off')  # Hide the axes
    plt.show()
    return AF_image


def registration(fixed_image, moving_image, parameter_map):
    
    FixedImage = sitk.GetImageFromArray(fixed_image)
    MovingImage = sitk.GetImageFromArray(moving_image)
    
    ParameterMap = sitk.GetDefaultParameterMap(parameter_map)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(FixedImage)
    elastixImageFilter.SetMovingImage(MovingImage)
    elastixImageFilter.SetParameterMap(ParameterMap)
    elastixImageFilter.Execute()

    ResultImage = elastixImageFilter.GetResultImage()
    TransParameterMap = elastixImageFilter.GetTransformParameterMap()
    
    ResultArray = sitk.GetArrayFromImage(ResultImage)
    FixedArray = sitk.GetArrayFromImage(FixedImage)
    MovingArray = sitk.GetArrayFromImage(MovingImage)
    
    return ResultArray, FixedArray, MovingArray, TransParameterMap


def registration_plot(ResultArray, FixedArray, MovingArray):
    
    ResultArrayNorm = ResultArray/np.amax(ResultArray)
    FixedArrayNorm = FixedArray/np.amax(FixedArray)
    MovingArrayNorm = MovingArray/np.amax(MovingArray)
    
    plt.figure(figsize=(18,8))
    ax = plt.subplot(1, 5, 1)
    plt.imshow(FixedArray, cmap='Blues')
    plt.axis('off')
    plt.title('Fixed Image')
    ax = plt.subplot(1, 5, 2)
    plt.imshow(MovingArray, cmap='Reds')
    plt.axis('off')
    plt.title('Moving Image')
    ax = plt.subplot(1, 5, 3)
    plt.imshow(ResultArray, cmap='Reds')
    plt.axis('off')
    plt.title('Result Image')
    ax = plt.subplot(1, 5, 4)
    plt.imshow(FixedArrayNorm, cmap='Blues', alpha=0.8)
    plt.axis('off')
    plt.imshow(MovingArrayNorm, cmap='Reds', alpha=0.4)
    plt.axis('off')
    plt.title('Before optimization')
    ax = plt.subplot(1, 5, 5)
    plt.imshow(FixedArrayNorm, cmap='Blues', alpha=0.8)
    plt.axis('off')
    plt.imshow(ResultArrayNorm, cmap='Reds', alpha=0.4)
    plt.axis('off')
    plt.title ('After optimization')
    plt.show()


def transformation(images_processed_LA, MALDI_BM, TransParameterMap):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(TransParameterMap)
    Trans_LA = []
    for images in images_processed_LA:
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(images))
        transformixImageFilter.Execute()
        LA_transformed = (sitk.GetArrayFromImage(transformixImageFilter.GetResultImage()))*MALDI_BM
        Trans_LA.append(LA_transformed)
    return Trans_LA

def structural_similarity_index(array1, array2):
    
    ssi = ssim(array1, array2)
    
    return ssi


