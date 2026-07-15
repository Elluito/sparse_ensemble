# Taken from https://github.com/VijayKalmath/Spectral-Representations-for-Convolutional-Neural-Networks/blob/master/src/modules/spectral_pooling.py
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import math

class SpectralPooling:

    def __init__(self):
        pass

    def tf_fouriertransform(self,spatial_image):
        '''
        Performs Fast Fourier Transform on the input image.
        Shifts DC Component of Fourier image to the Center

        Inputs:
        1) Spatial image , Real value , dimensions [channels,length,width]

        Outputs:
        1) Spectral image , Complex value Array , dimensions [channels,length,width]
        '''

        # Convert it into Fourier Transform
        fourier_image = tf.signal.fft2d(spatial_image)

        # DC should be shifted to Center
        fourier_image = tf.signal.fftshift(fourier_image)

        return fourier_image

    def tf_inversefouriertransform(self,fourier_image):
        '''
        Shifts DC Component of Fourier image to the top.

        Performs InverseFast Fourier Transform on the input image.

        Inputs:
        1) Spectral image , Complex value Array , dimensions [channels,length,width]

        Outputs:
        1) Spatial image , Real value , dimensions [channels,length,width]

        '''
        # Shift DC back
        fourier_image = tf.signal.ifftshift(fourier_image)

        # Convert it back into Spatial Image
        spatial_image = tf.signal.ifft2d(fourier_image)

        return spatial_image


    def tf_normalizeimage(self,spatial_image):
        '''
        Normalize Image data to values between [0..255]

        Inputs:
        1) Spatial image [length,width,channels]

        Outputs:
        1) Normalized Spatial image [length,width,channels]

        '''
        if spatial_image.shape[0] == 1:
            channel_max = tf.reduce_max(spatial_image, axis=(1, 2))
            channel_min = tf.reduce_min(spatial_image, axis=(1, 2))
        else :
            channel_max = tf.reduce_max(spatial_image, axis=(0, 1))
            channel_min = tf.reduce_min(spatial_image, axis=(0, 1))

        normalized_spatialimage = tf.divide(spatial_image - channel_min,channel_max - channel_min)

        return normalized_spatialimage


    def tf_lowpassfilter(self,fourier_image,filter_shape):
        '''
        Builds Low Pass filter using Fourier_image size and pool_size for dimensionreduction

        Inputs:

        1) Spectral image , Complex value Array , dimensions [channels,length,width]
        2) filter_shape , 2D shape

        Outputs:

        1) Filtered image , Complex value Array , dimensions [channels,length,width]
        '''
        filter_shape = list(filter_shape)
        fourierimage_size = fourier_image.shape

        if (fourierimage_size[2] + filter_shape[0]) % 2 == 1 :
            filter_shape[0] += 1

        if (fourierimage_size[3] + filter_shape[1]) % 2 == 1 :
            filter_shape[1] += 1

        top = int((fourierimage_size[2]-filter_shape[0])//2)
        bottom = int(top + filter_shape[0])

        left = int((fourierimage_size[3]-filter_shape[1])//2)
        right = int( left + filter_shape[1])

        filtered_image = fourier_image[:,:,top:bottom,left:right]

        return filtered_image

    def tf_treatcornercases(self,filtered_fourier_image):

        self.corner_set=[(0,0)]

        height = filtered_fourier_image.shape[2]
        width  = filtered_fourier_image.shape[3]

        if height % 2 == 0 :
            self.corner_set.append((int(height/2),0))
            if width % 2 == 0 :
                self.corner_set.append((int(height/2),int(width/2)))

        if width % 2 == 0 :
            self.corner_set.append((0,int(width/2)))
        mask = np.ones(shape=filtered_fourier_image.shape[1:])

        for coordinate in self.corner_set:
            mask[:,coordinate[0],coordinate[1]] = 0

        filtered_fourier_image_real = tf.math.real(filtered_fourier_image)
        filtered_fourier_image_imag = tf.math.imag(filtered_fourier_image)
        filtered_fourier_image_imag = filtered_fourier_image_imag * mask

        filtered_fourier_image = tf.complex(filtered_fourier_image_real, filtered_fourier_image_imag)

        return filtered_fourier_image

    def spectral_pool(self,spatial_image,filter_shape=(5,5)):
        assert filter_shape[0] > 0 , "Pool Size cannot be lesser than 1" # denominator can't be 0

        # Convert it into Channels First
        spatial_imagechannelsfirst = tf.transpose(spatial_image,perm=[0,3,1,2])

        # Get Fourier Image
        fourier_image = self.tf_fouriertransform(tf.cast(spatial_imagechannelsfirst, tf.complex64))


        # Apply low pass filter on Fourier Image
        if filter_shape[0] > 1 :
            filtered_fourier_image = self.tf_lowpassfilter(fourier_image,filter_shape)
        else :
            filtered_fourier_image = fourier_image

        filtered_fourier_image=self.tf_treatcornercases(filtered_fourier_image)

        # Convert it back into spatial image
        filtered_spatial_image= self.tf_inversefouriertransform(filtered_fourier_image)

        filtered_spatial_image_channelslast = tf.abs(tf.transpose(filtered_spatial_image,perm=[0,2,3,1]))
        filtered_spatial_image = self.tf_normalizeimage(filtered_spatial_image_channelslast)

        filtered_fourier_image = tf.transpose(filtered_fourier_image,perm=[0,2,3,1])
        magnitude_spectrum = 20*np.log(np.abs(filtered_fourier_image) + 1e-45)
        magnitude_spectrum = magnitude_spectrum/(tf.reduce_max(magnitude_spectrum))

        return magnitude_spectrum,filtered_spatial_image


    def tf_lowpassfilter_poolsize(self,fourier_image,pool_size=4):
        '''
        Builds Low Pass filter using Fourier_image size and pool_size for dimensionreduction

        Inputs:

        1) Spectral image , Complex value Array , dimensions [channels,length,width]
        2) Pool Size

        Outputs:

        1) Filtered image , Complex value Array , dimensions [channels,length,width]
        '''
        fourierimage_size = fourier_image.shape
        filter_shape = pool_size
        lowpass = np.ones(shape=fourierimage_size, dtype=np.float32)

        distance_from_corner = math.ceil((fourierimage_size[2] - (fourierimage_size[2]/filter_shape)) / 2)

        lowpass[:,:,:distance_from_corner,:] = 0
        lowpass[:,:,-distance_from_corner: ,:] = 0

        distance_from_corner = math.ceil((fourierimage_size[3] - (fourierimage_size[3]/filter_shape)) / 2)

        lowpass[:,:,:,:distance_from_corner] = 0
        lowpass[:,:,:,-distance_from_corner:] = 0

        filtered_image = fourier_image * lowpass

        return filtered_image



    def utils_spectral_pool(self,spatial_image,pool_size=4):

        # Convert it into Channels First
        spatial_imagechannelsfirst = tf.transpose(spatial_image,perm=[0,3,1,2])

        # Get Fourier Image
        fourier_image = self.tf_fouriertransform(tf.cast(spatial_imagechannelsfirst, tf.complex64))

        # Apply low pass filter on Fourier Image
        if pool_size > 1 :
            filtered_fourier_image = self.tf_lowpassfilter_poolsize(fourier_image,pool_size)
        else :
            filtered_fourier_image = fourier_image

        filtered_fourier_image=self.tf_treatcornercases(filtered_fourier_image)

        # Convert it back into spatial image
        filtered_spatial_image= self.tf_inversefouriertransform(filtered_fourier_image)

        filtered_spatial_image_channelslast = tf.abs(tf.transpose(filtered_spatial_image,perm=[0,2,3,1]))
        filtered_spatial_image = self.tf_normalizeimage(filtered_spatial_image_channelslast)

        filtered_fourier_image = tf.transpose(filtered_fourier_image,perm=[0,2,3,1])
        magnitude_spectrum = 20*np.log(np.abs(filtered_fourier_image) + 1e-45)
        magnitude_spectrum = magnitude_spectrum/(tf.reduce_max(magnitude_spectrum))

        return magnitude_spectrum,filtered_spatial_image
