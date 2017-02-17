"""
author: Melanie Daeschinger
description: Prepare data from data science bowl 2017 for analysis.
    First do some preprocessing steps. Then segment the lung tissue.
"""

import numpy as np
import Preprocessing
import Segmentation
import Loader
import os
import matplotlib.pyplot as plt
import time


# Time tracking
time_start = time.time()

# Constants
INPUT_FOLDER = os.getcwd() + '/input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# 1. PREPROCESSING
print("\n", "PREPROCESSING SAMPLE DATA (20 Patients)")

# 1. For each patient Load the whole stack of CT-Data, convert greyscale to HU and resample data
images = []
for i in range(len(patients)):
    ctData = Preprocessing.load_scan(INPUT_FOLDER + patients[i])
    # Convert greyscale to Hounsfield Units (HU)
    pixelValues = Preprocessing.get_pixels_hu(ctData)
    # Resample voxels to 1x1x1 distance
    resampled, spacing = Preprocessing.resample(pixelValues, ctData, [1, 1, 1])
    # print("Shape before resampling\t", pixelValues.shape)
    #print("Shape after resampling\t", resampled.shape)
    images.append(resampled)

patientImages = np.array(images)



# 2. SEGMENTATION OF LUNG TISSUE WITH WATERSHED
print("\n", "SEGMENTATION OF LUNG TISSUE WITH WATERSHED")

# 2.1.a For each patient Iterate over all Slices to segment Lung with Watershed
segmentedLungs= []
for i in range(len(patientImages)):
    print("Patient ", i )
    patient = patientImages[i]
    rows, cols = patient[0].shape
    segmentedImages = np.empty((len(patient),rows, cols))
    for n in range(len(patient)):
        segmentedLung, lungfilterArea, outline, watershedImage, sobelGradient, markerInternal, \
        markerExternal, markerWatershed = Segmentation.seperate_lungs(patientImages[i][n])
        segmentedImages[n, :, :] = segmentedLung
    segmentedLungs.append(segmentedImages)


# 3. SAVE ALL SEGMENTED LUNGS AS .NPY
print("\n", "SAVE ALL SEGMENTED LUNGS AS .NPY FILE")

# TODO Save all patients
for i in range(len(segmentedLungs)):
    Loader.save_stack(segmentedLungs[i], ('seg' + str(i)))



# 2.1.d OPTIONAL: Plot the segmented lung from one patient and the HU values
#Preprocessing.plot_3d(filtered, -500)

#plt.hist(segmentedLungs[0], bins=80, color='c')
#plt.xlabel("Hounsfield Unit [HU]")
#plt.ylabel("Frequency")
#plt.show()


time_end = time.time()
print("FULL PREPARATION TIME IN MINUTES= ", round((time_end - time_start)/60,2))