# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:26:38 2021

@author: Wenxi
"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from bm3d import bm3d
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from bm3d_demos.experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from bm3d_demos.guassian_param_selection import guassian_param_selection
from bm3d_demos.bm3d_param_selection import bm3d_param_selection
import cv2
from skimage import io, img_as_float
from skimage.filters import gaussian
import pandas as pd
from pathlib import Path


def run_model_eval():
    num_test_images = 10
    
    # base model parameter
    sigma = 0.018315638#bm3d_param_selection()['param']#0.1#bm3d_param_selection()['param']
    sigma_guassian_x = guassian_param_selection()['param'][0]
    sigma_guassian_y = guassian_param_selection()['param'][1]
    guassian_kernal = guassian_param_selection()['param'][2]
    
    #dl model outputs
    p = Path(r'project_model/results/test/test_latest.mat')    
    outputs = sio.loadmat(p.resolve())    
    
    print('path imported')
    base_psnrs = []
    base_maes = []
    base_mses = []
    
    guassian_psnrs = []
    guassian_maes = []
    guassian_mses = []
    
    dl_psnrs = []
    dl_maes = []
    dl_mses = []
    
    image_sample = []
    
    for i in range(num_test_images):
        orig_image = outputs['real_B'][i]
        noisy_img = outputs['real_A'][i]
        
        #test base model
        y_pred= bm3d(noisy_img, sigma)
    
        psnr = get_psnr(orig_image, y_pred)
        base_psnrs.append(psnr)
        
        mae = mean_absolute_error(orig_image, y_pred)
        base_maes.append(mae)
        
        mse = mean_squared_error(orig_image, y_pred)
        base_mses.append(mse)
        
        #test guassian model
        y_pred_guassian = cv2.GaussianBlur(noisy_img,
                                           ksize = (guassian_kernal,guassian_kernal),
                                           sigmaX = sigma_guassian_x,
                                           sigmaY = sigma_guassian_y,
                                           borderType = cv2.BORDER_DEFAULT)
        
        psnr = get_psnr(orig_image, y_pred_guassian)
        guassian_psnrs.append(psnr)
        
        mae = mean_absolute_error(orig_image, y_pred_guassian)
        guassian_maes.append(mae)
        
        mse = mean_squared_error(orig_image, y_pred_guassian)
        guassian_mses.append(mse)
    
        
        #test dl model
        y_pred_dl =  outputs['fake_B'][i]
    
        psnr_dl = get_psnr(orig_image, y_pred_dl)
        dl_psnrs.append(psnr_dl)
        
        mae_dl = mean_absolute_error(orig_image, y_pred_dl)
        dl_maes.append(mae_dl)
        
        mse_dl = mean_squared_error(orig_image, y_pred_dl)
        dl_mses.append(mse_dl)
        
        if i==0:
            image_sample +=[y_pred, y_pred_guassian, y_pred_dl]
        
        
    base_psnr = np.array(base_psnrs).mean()
    base_mae = np.array(base_maes).mean()
    base_mse = np.array(base_mses).mean()
    
    guassian_psnr = np.array(guassian_psnrs).mean()
    guassian_mae = np.array(guassian_maes).mean()
    guassian_mse = np.array(guassian_mses).mean()
    
    dl_psnr = np.array(dl_psnrs).mean()
    dl_mae = np.array(dl_maes).mean()
    dl_mse = np.array(dl_mses).mean()
    
    print(f'Evaluation Results on {num_test_images} images')
    print(f'\n******\nThe base line model evaluation results are:\n1.PSNR: {base_psnr}\n2.MAE: {base_mae}\n3.MSE: {base_mse}')
    print(f'\n******\nThe base line model evaluation results are:\n1.PSNR: {guassian_psnr}\n2.MAE: {guassian_mae}\n3.MSE: {guassian_mse}')
    print(f'\n******\nThe DL model evaluation results are:\n1.PSNR: {dl_psnr}\n2.MAE: {dl_mae}\n3.MSE: {dl_mse}')
    
    return base_psnrs, guassian_psnrs, dl_psnrs, base_maes, guassian_maes, dl_maes, base_mses, guassian_mse, dl_mses, image_sample

#run_model_eval()