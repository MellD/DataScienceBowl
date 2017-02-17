"""
author: Melanie Daeschinger
description: Preprocessing methods for CT-Scans.
external code: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
"""

import numpy as np
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pylab

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral) #estimate_sigma)
from skimage import measure
from skimage.segmentation import clear_border
from matplotlib import pyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


# Load the scans in the given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


# Convert greyscale to HU: (Greyscale * Rescale Slope) + Intercept
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # NOTE: Do not set the outside pixels to 0. It's better for segmentation later
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


# Pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ),
# which can hurt performance of CNN approaches. We can deal with this by isomorphic resampling
# Resample everything to 1mm x 1mm x 1mm
def resample(image, scan, new_spacing):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing



# Visualization Algorithm "Marching Cubes"
# @threshold: Plot certain structures. [-300, -400]: Plot lung tissue
def plot_3d(image, threshold):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# @variable footprint: Neighborhood
# @variable size: if no footprint is given, with ie size=2 := footprint=np.ones(2,2,2)
# @variable mode: Default is 'reflect'.
# @variable weight: The greater weight, the more denoising
def noise_reduction(image):
    #image = scipy.ndimage.filters.gaussian_filter(image, sigma)    ->no
    #image = scipy.ndimage.filters.gaussian_laplace(image, sigma=1) ->no
    #image = scipy.ndimage.filters.median_filter(denoised_images, size=2) #--> Segmentierter Bereich vergrößert sich-> No!
    #denoised_images = denoise_bilateral(image,sigma_color=None, sigma_spatial=10, multichannel=False) #-> No

    # remove artifacts connected to image border
    #TODO: Need better method for artifact removal
    #rows, cols = images[0].shape
    #clearedImages = np.empty((len(images), rows, cols))
    #for i in range(len(images)):
    #    clearedImages[i,:,:] = clear_border(images[i], bgval=-2000)

    # Estimate the average noise standard deviation across color channels.
    #sigma_est = estimate_sigma(images, multichannel=False, average_sigmas=True)
    #print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))


    denoised_images = denoise_tv_chambolle(image, weight=0.1, multichannel=False) # Choose weight not too big!

    return denoised_images


def print_pointcloud(images, interval_begin=-1000, interval_end=165):
    fig = pylab.figure()
    ax = Axes3D(fig)

    #Get Coordinates where HU values are between a certain interval
    ii = np.where((images >= interval_begin) & (images <= interval_end))
    # ii = (array([0,0,0, ..., 334, 334, 334], dtype=int64),
    #       array([ 84,  84,  84, ..., 170, 170, 170], dtype=int64),
    #       array([233, 234, 235, ...,  99, 100, 101], dtype=int64))

    z_vals = ii[0]
    x_vals = ii[1]
    y_vals = ii[2]

    print(len(y_vals))
    #For >= -1000: 6386957 points
    #For >= -1000 & <= 165: 6263747 points
    #For >= -800:  1862225 points
    #For >= -20:    523477 points

    #Remove every second point from list (because of performance issues)
    z_vals = np.delete(z_vals, np.arange(0,z_vals.size,2))
    x_vals = np.delete(x_vals, np.arange(0,x_vals.size,2))
    y_vals = np.delete(y_vals, np.arange(0,y_vals.size,2))

    z_vals = np.delete(z_vals, np.arange(0,z_vals.size,2))
    x_vals = np.delete(x_vals, np.arange(0,x_vals.size,2))
    y_vals = np.delete(y_vals, np.arange(0,y_vals.size,2))

    #Plot all points:
    ax.scatter(y_vals, x_vals, z_vals)
    pyplot.show()


#TODO: Noise reduction
#TODO: Raise morphologies


