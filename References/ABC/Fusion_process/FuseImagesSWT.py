# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:21:50 2022

@author: denis
"""

import cv2
import pywt
import numpy as np


def check_and_pad(img, lvl):
    val = 2**lvl
    width_diff = img.shape[1] % val
    height_diff = img.shape[0] % val

    print(val, height_diff, width_diff)
    if width_diff > 0:
        img = np.pad(img, [(0, 0), (0, val - width_diff)], mode='constant')
    if height_diff > 0:
        img = np.pad(img, [(0, val - height_diff), (0, 0)], mode='constant')

    return img


class FuseImagesSWT:
    available_methods = ['mean', 'min', 'max']
    
    def __init__(self, wavelet_function = 'db1', method = 'mean', level = 2):
        self.lvl = level

        if wavelet_function in pywt.wavelist():
            self.wavelet_function = wavelet_function
        else:
            self.wavelet_function = 'db1'
        
        if method in self.available_methods:
            self.method = method
        else:
            self.method = 'mean'
    
    def setMethod(self, method):
        if method in self.available_methods:
            self.method = method

    def setWavelet(self, wavelet):
        if wavelet in pywt.wavelist():
            self.wavelet_function = wavelet
    
    def fuseCoefficient(self, c1, c2):
        if (self.method == 'mean'):
            coeff = (c1 + c2) / 2
        elif (self.method == 'min'):
            coeff = np.minimum(c1, c2)
        elif (self.method == 'max'):
            coeff = np.maximum(c1, c2)
        
        return coeff
    
    def fuseImages(self, ir_img, vi_img):
        # Check if images are already converted to Gray, otherwise convert them
        if ir_img.shape[2] > 1:
            gray_ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_ir_img = ir_img
            
        if vi_img.shape[2] > 1:
            gray_vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_vi_img = vi_img

        print(gray_ir_img.shape)
        print(gray_vi_img.shape)

        gray_ir_img = check_and_pad(gray_ir_img, self.lvl)
        gray_vi_img = check_and_pad(gray_vi_img, self.lvl)

        print(gray_ir_img.shape)
        print(gray_vi_img.shape)

        # Perform wavelet transform on each image
        coeff1 = pywt.swt2(gray_ir_img[:,:], self.wavelet_function, level = self.lvl)
        coeff2 = pywt.swt2(gray_vi_img[:,:], self.wavelet_function, level = self.lvl)

        # Fuse coefficients for each level in both images
        fusedCoeff = []
        
        for i in range(self.lvl - 1):
            tempCoff = []
            # Fuses the approximation values of the top level decomposition
            tempCoff.append(self.fuseCoefficient(coeff1[i][0], coeff2[i][0]))
            
            # Fuses the other 3 coefficents
            c1 = self.fuseCoefficient(coeff1[i][1][0], coeff2[i][1][0])
            c2 = self.fuseCoefficient(coeff1[i][1][1], coeff2[i][1][1])
            c3 = self.fuseCoefficient(coeff1[i][1][2], coeff2[i][1][2])

            tempCoff.append((c1, c2, c3))
            fusedCoeff.append(tempCoff)

        # Reconstruct the image by transforming it back to get the fused image
        fusedImage = pywt.iswt2(fusedCoeff, self.wavelet_function)

        # Normalizing the values
        fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),
                                           (np.max(fusedImage) - np.min(fusedImage))), 255)

        return fusedImage.astype(np.uint8)