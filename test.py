# -*- coding: utf-8 -*-
"""
Modified at Dec 16 2023
"""

import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io as sio
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import PIL
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import tensorflow as tf
layers = tf.keras.layers
from include.my_circular_layer import Conv2D_circular
import include.utils as vf
# from scipy.ndimage.filters import convolve, median_filter
from scipy.ndimage import convolve, median_filter
# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
from train.try_7.decoder import Decoder
from train.try_7.encoder import Encoder


def buildModel(model_path, patch_rows=32, patch_cols=32, channels=1, block_size=8, use_circular=True):

    
    w_rows = int((patch_rows) / block_size)
    w_cols = int((patch_cols) / block_size)
    
    input_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_img')
    input_strenght_alpha = layers.Input(shape=(1,), name='strenght_factor_alpha')
    input_watermark = layers.Input(shape=(w_rows, w_cols, 1), name='input_watermark')
    
    # Rearrange input 
    rearranged_img = l1 = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='rearrange_img')(input_img)
    
    dct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct1')
    dct_layer2 = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='dct2')
    idct_layer = layers.Conv2D(64, (1, 1), activation='linear', padding='same', use_bias=False, trainable=False, name='idct')
    dct_layer_img = dct_layer(rearranged_img)
    # Encoder
    encoder_model = Encoder()([dct_layer_img, input_watermark])
    encoder_model = idct_layer(encoder_model)
    # Strength
    encoder_model = tf.math.multiply(encoder_model, input_strenght_alpha, name="strenght_factor")
    encoder_model = layers.Add(name='residual_add')([encoder_model, l1])
    encoder_model = x = layers.Lambda(tf.nn.depth_to_space, arguments={'block_size':block_size}, name='enc_output_depth2space')(encoder_model)
    
    # Attack (The attacks occure in test phase)
    
    # Watermark decoder
    input_attacked_img = layers.Input(shape=(patch_rows, patch_cols, 1), name='input_attacked_img')
    decoder_model = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size':block_size}, name='dec_input_space2depth')(input_attacked_img)
    decoder_model = dct_layer2(decoder_model)
    decoder_model = Decoder()(decoder_model)
    decoder_model = layers.Conv2D(1, (1, 1), dilation_rate=1, activation='sigmoid', padding='same', name='dec_output_depth2space')(decoder_model)
    
    # Whole model
    embedding_net = tf.keras.models.Model(inputs=[input_img, input_watermark, input_strenght_alpha], outputs=[x])
    extractor_net = tf.keras.models.Model(inputs=[input_attacked_img], outputs=[decoder_model])
    
    # Set weights
    DCT_MTX = sio.loadmat('./transforms/DCT_coef.mat')['DCT_coef']
    dct_mtx = np.reshape(DCT_MTX, [1,1,64,64])
    embedding_net.get_layer('dct1').set_weights(np.array([dct_mtx]))
    extractor_net.get_layer('dct2').set_weights(np.array([dct_mtx]))
    
    IDCT_MTX = sio.loadmat('./transforms/IDCT_coef.mat')['IDCT_coef']
    idct_mtx = np.reshape(IDCT_MTX, [1,1,64,64])
    embedding_net.get_layer('idct').set_weights(np.array([idct_mtx]))
    
    embedding_net.load_weights(model_path,by_name = True)
    extractor_net.load_weights(model_path,by_name = True)
    return embedding_net, extractor_net

# %%
# Test images
img_rows, img_cols = 512, 512
test_folder = './images/{}x{}'.format(img_rows, img_cols)
test_imgs_files = [f for f in listdir(test_folder) if isfile(join(test_folder, f)) and (f.endswith('.bmp') or f.endswith('.gif'))]

# Exp Info
exp_id = 'new_model'
use_circular = True
save_samples = True
patch_rows, patch_cols = 32, 32
block_size = 8
Is_mean_normalized = True
mean_normalize = 128.0
std_normalize = 255.0

assert patch_rows == patch_cols, 'Patches must have same rows and columns'
assert img_rows % patch_rows == 0 and img_cols % patch_cols == 0, 'Image size must be dividable by the patch size'

print('Analyzing Experiment {}...'.format(exp_id))
time.sleep(1)

# Log folder
analysis_folder = './logs/{}/Analysis'.format(exp_id)
if os.path.exists(analysis_folder) == False:
    os.mkdir(analysis_folder)

sampled_embeded_folder = os.path.join(analysis_folder, 'Sampled Embeddings')
if os.path.exists(sampled_embeded_folder) == False:
    os.mkdir(sampled_embeded_folder)
    
sampled_attack_folder = os.path.join(analysis_folder, 'Sampled Attacks')
if os.path.exists(sampled_attack_folder) == False:
    os.mkdir(sampled_attack_folder)

# Model Definition
model_path = './logs/{}/Weights/weights_final.h5'.format(exp_id)
embedding_net, extractor_net = buildModel(model_path,use_circular=use_circular)

# List of Attacks
# Rotation
def rotate(img, Q): # Q = rotation degree
    im = PIL.Image.fromarray(img)
    rotated_img = im.rotate(Q, resample=Image.BILINEAR)
    return np.array(rotated_img)

#JPEG
def jpg(img, Q):
    global exp_id
    cv2.imwrite('temporary_files/temp_{}.jpg'.format(exp_id), img, [cv2.IMWRITE_JPEG_QUALITY, int(Q)])
    Iw_attacked = cv2.imread('temporary_files/temp_{}.jpg'.format(exp_id), cv2.IMREAD_GRAYSCALE)
    return Iw_attacked


attacks_list = [
                {'func':jpg, 'name':'JPEG', 'params': np.array([90, 70, 60, 50, 30]), 'active':True}, # JPEG
                ] 

# %%
# Expriments
alpha_values = np.array(range(1, 5+1, 1)) / (5.0)
num_random_watermarks = 1

w_rows = int((patch_rows) / block_size)
w_cols = int((patch_cols) / block_size)
bits_per_patch = w_rows * w_cols
n_patches = int((img_rows * img_cols) / (patch_rows * patch_cols))
total_cap = n_patches * bits_per_patch

message_length = 1024
analysis_folder = os.path.join(analysis_folder, '{}-Bits'.format(message_length))
if os.path.exists(analysis_folder) == False:
    os.mkdir(analysis_folder)
assert total_cap % message_length == 0, 'Total Capacity must be dividable by message length'
n_redundancy = total_cap // message_length

# %%
psnr_means = []
psnr_stds = []
ssim_means = []

# Computing PSNRs
print('Computing PSNRs...')

for test_img in test_imgs_files:
    print('\tProcessing ', test_img, ' ...')
    im = plt.imread(os.path.join(test_folder, test_img))
#    if im == None:
#        print('\t[!] Error loading ', test_img)
#        continue
    if im.shape[-1] == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im
    
    # Normalize image
    Im_normalized = (im_gray.copy() - mean_normalize if Is_mean_normalized else 0) / std_normalize
    
    num_batch = (img_rows * img_cols) // (patch_rows * patch_cols)
    psnr_values_per_alpha_mean = []
    psnr_values_per_alpha_std = []
    ssim_values_per_alpha_mean = []
    
    for alpha in alpha_values:    
        # Compute PSNRs
        tmp_psnr = []
        tmp_ssim = []
        for n in range(num_random_watermarks ):
            Im_32x32_patchs = vf.partitioning(Im_normalized, p_size=patch_rows)
            W = np.random.randint(low=0, high=2, size=(num_batch, w_rows, w_cols, 1)).astype(np.float32)
            # Apply embedding network
            # Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, alpha*np.ones_like(W)])
            Iw_batch = embedding_net([Im_32x32_patchs, W, alpha * np.ones_like(W)], training=False)
            # reconstruct Iw
            Iw = vf.tiling(Iw_batch, rec_size=img_rows)
            Iw *= std_normalize
            Iw += mean_normalize if Is_mean_normalized else 0
            Iw[Iw > 255] = 255
            Iw[Iw < 0] = 0
            Iw = np.uint8(Iw.squeeze())
            # PSNR
            #psnr = 10*np.log10(255**2/np.mean((im_gray - Iw)**2))
            psnr = peak_signal_noise_ratio(im_gray, Iw, data_range=255)
            tmp_psnr.append(psnr)
            # SSIM
            tmp_ssim.append(structural_similarity(im_gray, Iw, win_size=9, data_range=255))
            
            # Save sample image
            if n == 0 and save_samples == True:
                cv2.imwrite(os.path.join(sampled_embeded_folder, '{}_[{}].png'.format(test_img[:-4], alpha)), Iw)
        
        psnr_values_per_alpha_mean.append(np.mean(tmp_psnr))
        psnr_values_per_alpha_std.append(np.std(tmp_psnr))
        ssim_values_per_alpha_mean.append(np.mean(tmp_ssim))
    
    psnr_means.append(psnr_values_per_alpha_mean)
    psnr_stds.append(psnr_values_per_alpha_std)
    ssim_means.append(ssim_values_per_alpha_mean)
 
# %% 
print('Computing BER....')

bers_per_attacks = []
for attack in attacks_list: # for all attacks
    sampled_folder_per_attack = os.path.join(sampled_attack_folder, attack['name'])
    if os.path.exists(sampled_folder_per_attack) == False:
        os.mkdir(sampled_folder_per_attack)

    if attack['active'] == False:
        continue
    
    print('\tPerforming {} attack...'.format(attack['name']))
    
    bers_per_attack_params = []
    for attack_params in tqdm(attack['params']): # for all attack params

        bers_per_attack_per_image = []
        bers_per_attack_per_image_robust = []
        for test_img in test_imgs_files:  # for all images
            im = plt.imread(os.path.join(test_folder, test_img))
            if im.shape[-1] == 3:
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            else:
                im_gray = im
            
            # Normalize image
            Im_normalized = (im_gray.copy() - mean_normalize if Is_mean_normalized else 0) / std_normalize
            
            bers_per_alpha = []
            
            for alpha in alpha_values:
                tmp_bers = []
                for n in range(num_random_watermarks):
                    Im_32x32_patchs = vf.partitioning(Im_normalized, p_size=patch_rows)

                    #W = np.random.randint(low=0, high=2, size=(n_patch, w_rows, w_cols, 1)).astype(np.float32)
                    W = np.random.randint(low=0, high=2, size=(message_length,1)).astype(np.float32)
                    W = np.reshape(W, [-1, w_rows, w_cols])

                    #Redundant embedding
                    N_mtx = (img_rows//patch_rows) # assuming img_rows == img_cols
                    W_robust = np.zeros([N_mtx, N_mtx, w_rows, w_cols], dtype=np.float32)
                    k = 0
                    n_repeats = 0
                    for d in range(N_mtx):
                        for i in range(N_mtx):
                            if k >= W.shape[0]:
                                break
                            W_robust[i, (i+d)%N_mtx, :, :] = W[k, :, :] ##### % 8 for 256x256
                            n_repeats += 1
                            if n_repeats >= n_redundancy:
                                n_repeats = 0
                                k += 1
                                
                    W_robust = np.reshape(W_robust, [-1, w_rows, w_cols, 1])

                    # Apply embedding network
                    # Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W_robust, alpha*np.ones_like(W_robust)])
                    Iw_batch = embedding_net([Im_32x32_patchs, W_robust, alpha * np.ones_like(W_robust)], training=False)

                    # reconstruct Iw
                    Iw = vf.tiling(Iw_batch, rec_size=img_rows)
                    Iw *= std_normalize
                    Iw += mean_normalize if Is_mean_normalized else 0
                    Iw[Iw > 255] = 255
                    Iw[Iw < 0] = 0
                    Iw = np.uint8(Iw.squeeze())
                    
                    # Apply Attack
                    Iw_attacked = attack['func'](Iw, attack_params)

                    Iw_tmp = Iw_attacked
                    
                    Iw_attacked = (Iw_attacked - mean_normalize if Is_mean_normalized else 0) / std_normalize
                    
                    
                    Iw_attacked_patchs = vf.partitioning(Iw_attacked, p_size=patch_rows)
        
                    # Feed to extractor
                    # w_batch = extractor_net.predict_on_batch([Iw_attacked_patchs])
                    w_batch = extractor_net([Iw_attacked_patchs], training=False)
                    w_batch = w_batch > 0.5

                    # Majority voting
                    w_batch = np.reshape(w_batch, [N_mtx, N_mtx, w_rows, w_cols])
                    w_extracted = np.zeros_like(W)
                    
                    k = 0
                    n_repeats = 0
                    
                    for d in range(N_mtx):
                        for i in range(N_mtx):
                            if k >= w_extracted.shape[0]:
                                break
                            w_extracted[k, :, :] += w_batch[i, (i+d)%N_mtx, :, :] ####### % 8 for 256x256 ?
                            n_repeats += 1
                            if n_repeats >= n_redundancy:
                                n_repeats = 0
                                k += 1
                    
                    w_extracted = (w_extracted > n_redundancy//2)

        
                    # Compute BER
                    xor_w = (W != w_extracted)
                    ber = np.sum(xor_w) / (message_length)
                    tmp_bers.append(ber)
                    
                    # Same sample
                    if n == 0 and alpha == 1.0 and save_samples == True:
                        file_name = '{}_{}_{:.2f}_{:.3f}.png'.format(attack['name'], test_img[:-4], attack_params, ber)
                        Iw_tmp = np.uint8(Iw_tmp)
                        cv2.imwrite(os.path.join(sampled_folder_per_attack, file_name), Iw_tmp)
                
                bers_per_alpha.append(np.mean(tmp_bers))
            
            bers_per_attack_per_image.append(bers_per_alpha)
        bers_per_attack_params.append(bers_per_attack_per_image)
    bers_per_attacks.append(np.array(bers_per_attack_params))

bers_array = bers_per_attacks # [attack, attack_param, images, alpha]

# Save Results in a mat file
print('Saving Results...')
var_dict = {}


var_dict['psnr_means'] = psnr_means
var_dict['psnr_std'] = psnr_stds
var_dict['ssim_means'] = ssim_means

attack_idx = - 1
for idx, attack in tqdm(enumerate(attacks_list)):
    if attack['active'] == False:
        continue
    attack_idx += 1
    var_dict[ attack['name'] ] = bers_array[attack_idx]

var_dict['images'] = test_imgs_files
sio.savemat(os.path.join(analysis_folder, 'ber_analysis.mat'), mdict=var_dict)

# %% Plotting
# Plot PSNR
plt.figure(figsize=(8,4))
for i in range(len(psnr_means)):
    plt.errorbar(x=alpha_values, y=psnr_means[i], yerr=1*np.array(psnr_stds[i]), marker='^')

#plt.xticks(np.arange(len(alpha_values)), alpha_values)
plt.xlabel('Strength Factor(Alpha)')
plt.ylabel('PSNR')
plt.title('PSNR Per Alpha')
plt.legend(test_imgs_files)
plt.savefig(os.path.join(analysis_folder, 'PSNR.png'))

# Plot BERs per Attack
attack_idx = -1
for i, attack in enumerate(attacks_list):
    if attack['active'] == False:
        continue
    attack_idx += 1
    print('Plotting {}...'.format(attack['name']))
    tmp_ber_array = bers_array[attack_idx]
    mean_bers = np.mean(tmp_ber_array, axis=1)
    
    # 3D Plot
    fig = plt.figure(figsize=(8,4))
    ax = Axes3D(fig)
    X, Y = np.meshgrid(alpha_values, attack['params'])
    ax.plot_surface(X , Y, mean_bers, cmap=cm.coolwarm, antialiased=True)
    ax.set_xlabel('Strength Factor')
    ax.set_ylabel('Parameters')
    ax.set_zlabel('BER')
    ax.set_title(attack['name'])
    fig.savefig(os.path.join(analysis_folder, '{}_3D.png'.format(attack['name'])))
    
    # 2D Line plot
    selected_params = sorted(np.random.permutation(len(attack['params']))[:4])
    plt.figure(figsize=(8,4))
    plt.plot(mean_bers.T[:, selected_params])
    plt.xticks(np.arange(len(alpha_values)), alpha_values)
    plt.xlabel('Strength Factor(Alpha)')
    plt.ylabel('BER')
    plt.legend(attack['params'][selected_params])
    plt.title(attack['name'])
    plt.savefig(os.path.join(analysis_folder, '{}_2D_sampled.png'.format(attack['name'])))
    
#impulse responce for a flat image
Im = np.ones((256,256)) * 0.5
Im_32x32_patchs = vf.partitioning(Im, p_size=32)
W = np.zeros((64,4,4,1))
Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, np.ones_like(W)])
Iw_zeros = vf.tiling(Iw_batch ,256)
W[:,1,1,:] = 1
Iw_batch = embedding_net.predict_on_batch([Im_32x32_patchs, W, np.ones_like(W)])
Iw_impulse = vf.tiling(Iw_batch ,256)
Impulse_responce = Iw_impulse-Iw_zeros

plt.figure(figsize=(8,4))
plt.imshow(Impulse_responce[0:32,0:32],cmap=cm.inferno)
plt.colorbar()
plt.savefig(os.path.join(analysis_folder, '{}.png'.format('Impulse_responce')))
