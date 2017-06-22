
# 
#  preprocess_img.py
#  morphe-mu
#  
#  Created by Ilya Mikhaltsou on 2017-04-23.
#  Copyright 2017 Ilya Mikhaltsou. All rights reserved.
# 

import skimage
import skimage.color
import skimage.transform
import skimage.morphology
import skimage.draw
import numpy as np

def make_luv(img):
    new = skimage.color.rgb2luv(img)
    l = np.dot(new, np.array([0.01, 0, 0]))
    u = np.dot(new, np.array([0, 0.01, 0]))
    v = np.dot(new, np.array([0, 0, 0.01]))
    return (l, u, v)
    
def higher_contrast(l, u, v):
    grad_L = skimage.filters.sobel(l)
    grad_U = skimage.filters.sobel(u)
    grad_V = skimage.filters.sobel(v)

    init1 = np.zeros((20,20))
    init1[skimage.draw.circle(10, 10, 5)] = 1
    init2 = np.zeros((20,20))
    init2[skimage.draw.circle(10, 10, 8)] = 1
    def m_closing(img, s1, s2):
        return skimage.morphology.dilation(1*skimage.morphology.erosion(img, s1)**1.9, s2)

    grad_LUV = m_closing(grad_L, init1, init2)**1.0 + \
                0.3*m_closing(grad_U, init1, init2)**1.0 + \
                0.3*m_closing(grad_V, init1, init2)**1.0
    
    grad_LUV = skimage.filters.gaussian(grad_LUV, sigma = 1.)
    grad_LUV = grad_LUV / np.max(grad_LUV)
    
    l_updated = 1 / ( 1 + np.exp(1 * (22*grad_LUV+1.5) * (0.4 - l)))
    return l_updated
    
def make_rgb(l, u, v):
    img_luv = 100*np.stack((l,u,v), axis = 2)
    return skimage.color.luv2rgb(img_luv)
    
def downsample(img, factor):
    return skimage.transform.pyramid_reduce(img, factor, multichannel=(len(img.shape) == 3 and img.shape[2] == 3))
    
