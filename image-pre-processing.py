
# coding: utf-8

# In[7]:

import os
from os.path import basename
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_each_array(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0/(maxval-minval))
        X = arr[:, i]
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    return arr

def normalize(filein):
    img = Image.open(filein).convert('RGBA')
    arr = np.array(img)
    return Image.fromarray(normalize_each_array(arr).astype('uint8'), 'RGBA').convert('RGB')

def resize(filename):
    oldImage = Image.open(filename)
    old_size = oldImage.size  # old_size[0] is in (width, height) format
    desired_size = 128
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = oldImage.resize(new_size, Image.ANTIALIAS)
    white_color = "#ffffff"
    new_im = Image.new("RGB", (desired_size, desired_size), white_color)
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2)) 
    return new_im

def normalize_and_resize(filename, savefilename, savedir):
    #step1: resize
    
    new_im = resize(filename)
    savedir = savedir.replace("converted_images", "converted_images_resized")
    makePathIfNotExist(savedir)
    new_im.save(savedir +"/"+ savefilename + ".png", "PNG")
   #step2: normalize
    new_im = normalize(savedir+"/"+savefilename+".png")
    savedir = savedir.replace("converted_images_resized", "converted_images_normalized")
    makePathIfNotExist(savedir)
    new_im.save(savedir +"/"+ savefilename + ".png", "PNG")   



def makePathIfNotExist(path):
    if not os.path.exists(path):
           os.makedirs(path)

def main():
   rootDir = '/home/maysam.mokarian/notebooks/files/gear_images'
   saveDir = '/home/maysam.mokarian/notebooks/files/converted_images'
   for dirName, subdirList, fileList in os.walk(rootDir):
       print('Found directory: %s' % dirName)
       print("coverting dir: " + saveDir + dirName.replace(rootDir, ""))
       savepath = saveDir + dirName.replace(rootDir, "")
       makePathIfNotExist(savepath)
   
       for fname in fileList:
           savedir = dirName.replace("gear_images", "converted_images")
           normalize_and_resize(dirName + "/" + fname, os.path.splitext(fname)[0], savedir)
main()

# In[13]:

import cv2
import numpy as np
from matplotlib import pyplot as plt

img0=cv2.imread('/home/maysam.mokarian/notebooks/files/gear_images/axes/100172.jpeg')
img1=cv2.imread('/home/maysam.mokarian/notebooks/files/converted_images_resized/axes/100172.png')
img2=cv2.imread('/home/maysam.mokarian/notebooks/files/converted_images_normalized/axes/100172.png')


print(img0.size)
print(img1.size)
print(img2.size)

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Subplot for original image
a=fig.add_subplot(1,3,1)
imgplot = plt.hist(img0.ravel())
a.set_title('Original')

# Subplot for processed image
a=fig.add_subplot(1,3,2)
imgplot = plt.hist(img1.ravel())
a.set_title('Resized')


# Subplot for processed image
a=fig.add_subplot(1,3,3)
imgplot = plt.hist(img2.ravel())
a.set_title('Normalized')

