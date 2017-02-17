"""
author: Melanie Daeschinger
description: Test methods and look for improval process
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


# Load one segmented Lung
segmentedLung = Loader.load_stack('test1.npy')

#Preprocessing.print_pointcloud(segmentedLung)



# 2.1.b OPTIONAL: Just show some example markers from the middle Slice (i.e picture 125)
#test_patient_internal, test_patient_external, test_patient_watershed = Segmentation.generate_markers(segmentedLung[100])
#print("Internal Marker")
#plt.imshow(test_patient_internal, cmap='gray')
#plt.show()
#print("External Marker")
#plt.imshow(test_patient_external, cmap='gray')
#plt.show()
#print("Watershed Marker")
#plt.imshow(test_patient_watershed, cmap='gray')
#plt.show()



# 2.1.c OPTIONAL: Lung Segmentation with Watershed for only one Slice
#test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, \
#test_marker_external, test_marker_watershed = Segmentation.seperate_lungs(patientImages[0][20])

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
#plt.imshow(segmentedLungs[0][100], cmap='gray')
#plt.show()


filteredLung = Preprocessing.noise_reduction(segmentedLung)

print("Noise reduction:")
plt.imshow(filteredLung[100], cmap='gray')
plt.show()

plt.hist(filteredLung.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Unit [HU]")
plt.ylabel("Frequency")
plt.show()

Preprocessing.print_pointcloud(filteredLung)

morph_extracts = Segmentation.morphological_extraction(segmentedLung)

print("Morph Extraction:")
plt.imshow(morph_extracts[100], cmap='gray')
plt.show()

plt.hist(morph_extracts.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Unit [HU]")
plt.ylabel("Frequency")
plt.show()

print("After Morph extracting:")
Preprocessing.print_pointcloud(filteredLung,1, 1000)



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


#Preprocessing.plot_3d(segmented_lungs_fill, 0)


time_end = time.time()
print("FULL PREPARATION TIME IN MINUTES= ", round((time_end - time_start)/60,2))