"""
author: Melanie Daeschinger
description: Prepare data from data science bowl 2017 for analysis.
    First do some preprocessing steps. Then segment the lung tissue.
"""

import numpy as np
import Preprocessing
import Segmentation
import os
import matplotlib.pyplot as plt
import time

#Time tracking
time_start = time.time()

#Constants
INPUT_FOLDER = os.getcwd() + '/input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# 1. PREPROCESSING
print("\n", "PREPROCESSING SAMPLE DATA (20 Patients)")

# 1. For each patient Load the whole stack of CT-Data, convert greyscale to HU and resample data
images = []
for i in range(1): #len(patients)):
    ctData = Preprocessing.load_scan(INPUT_FOLDER + patients[i])
    # Convert greyscale to Hounsfield Units (HU)
    pixelValues = Preprocessing.get_pixels_hu(ctData)
    # Resample voxels to 1x1x1 distance
    resampled, spacing = Preprocessing.resample(pixelValues, ctData, [1, 1, 1])
    # print("Shape before resampling\t", pixelValues.shape)
    print("Shape after resampling\t", resampled.shape)
    images.append(resampled)

patientImages = np.array(images)


# OPTIONAL
# 1.1. Mesh Generation with Marching Cubes
# Preprocessing.plot_3d(patientImages[1], -500)

#TODO: Filter noises

# 2.1. SEGMENTATION OF LUNG TISSUE WITH WATERSHED
print("\n", "SEGMENTATION OF LUNG TISSUE WITH WATERSHED")

# 2.1.a For each patient Iterate over all Slices to segment Lung with Watershed
segmentedLungs= []
for i in range(len(patientImages)):
    #segmentedImages = []
    patient = patientImages[i]
    rows, cols = patient[0].shape
    segmentedImages = np.empty((len(patient),rows, cols))
    for n in range(len(patient)):
        segmentedLung, lungfilterArea, outline, watershedImage, sobelGradient, markerInternal, \
        markerExternal, markerWatershed = Segmentation.seperate_lungs(patientImages[i][n])
        segmentedImages[n, :, :] = segmentedLung
    segmentedLungs.append(segmentedImages)


# 2.1.b OPTIONAL: Just show some example markers from the middle Slice (i.e picture 125)
# test_patient_internal, test_patient_external, test_patient_watershed = Segmentation.generate_markers(PatientImages.index(2)[125])
# print("Internal Marker")
# plt.imshow(test_patient_internal, cmap='gray')
# plt.show()
# print("External Marker")
# plt.imshow(test_patient_external, cmap='gray')
# plt.show()
# print("Watershed Marker")
# plt.imshow(test_patient_watershed, cmap='gray')
# plt.show()

# 2.1.c OPTIONAL: Lung Segmentation with Watershed for only one Slice
#test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, \
#test_marker_external, test_marker_watershed = Segmentation.seperate_lungs(patientImages[0][100])

#print("Sobel Gradient")
#plt.imshow(test_sobel_gradient, cmap='gray')
#plt.show()
#print("Watershed Image")
#plt.imshow(test_watershed, cmap='gray')
#plt.show()
#print("Outline after reinclusion")
#plt.imshow(test_outline, cmap='gray')
#plt.show()
#print("Lungfilter after closing")
#plt.imshow(test_lungfilter, cmap='gray')
#plt.show()
#print("Segmented Lung")
#plt.imshow(test_segmented, cmap='gray')
#plt.show()


# 2.1.d OPTIONAL: Plot the segmented lung from one patient and the HU values
#Preprocessing.plot_3d(segmentedLungs[0], -500)

plt.hist(segmentedLungs[0], bins=80, color='c')
plt.xlabel("Hounsfield Unit [HU]")
plt.ylabel("Frequency")
plt.show()


# 2.2. SEGMENTATION OF LUNG TISSUE WITH "STUDENTS"- METHOD

# 2.1.a Lung Segmentation with Students' method just for one patient
#segmented_lungs = Segmentation.segment_lung_mask(PatientImages[1], False)
#segmented_lungs_fill = Segmentation.segment_lung_mask(PatientImages[1], True)

#print("One Slice of Lung Mask ie 100")
#plt.imshow(segmented_lungs[100], cmap='gray')
#plt.show()
#print("One Slice of Filled Lung Mask ie 100 -> What we wanna use")
#plt.imshow(segmented_lungs_fill[100], cmap='gray')
#plt.show()

#plt.hist(segmented_lungs[100].flatten(), bins=80, color='c')
#plt.xlabel("Hounsfield Unit [HU]")
#plt.ylabel("Frequency")
#plt.show()

#Preprocessing.plot_3d(segmented_lungs_fill, 0)

time_end = time.time()
print("FULL PREPARATION TIME IN MINUTES= ", round((time_end - time_start)/60,2))