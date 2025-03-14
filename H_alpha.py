import astropy
from astropy.io import fits
import os
import numpy as np

def ping():
    return 'pong'

def dark():
    home_dir = os.getcwd()+'/data/Darks'
    files= os.listdir(home_dir)
    dark_names = [f for f in files if f.endswith(('.fits.gz'))]
    dark_files= [(fits.open(f'{home_dir}/{dark_names[i]}')[0]) for i in range(len(dark_names))]
    dark_data= [file.data for file in dark_files]

    return dark_files, dark_data

def bias():
    home_dir = os.getcwd()+'/data/Bias'
    files= os.listdir(home_dir)
    bias_names = [f for f in files if f.endswith(('.fits.gz'))]
    bias_files= [(fits.open(f'{home_dir}/{bias_names[i]}')[0]) for i in range(len(bias_names))]
    bias_data= [file.data for file in bias_files]

    return bias_files, bias_data

def lights():
    home_dir = os.getcwd()+'/data/Lights'
    files= os.listdir(home_dir)
    lights_names = [f for f in files if f.endswith(('.fits.gz'))]
    lights_files= [(fits.open(f'{home_dir}/{lights_names[i]}')[0]) for i in range(len(lights_names))]
    lights_data= [file.data for file in lights_files]

    return lights_files, lights_data

def average_image(image_list):
    return np.mean(image_list, axis=0)

def corrected_image(l,d_avg,b_avg):
    return l - d_avg-b_avg
