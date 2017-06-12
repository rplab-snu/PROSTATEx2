import dicom
import os
import csv
import time
import scipy.misc
import numpy as np
import skimage.feature as skf
import skimage.measure as skm
import cv2

# Image Sequence
ImageSequence = ["T2tra", "T2sag", "ADC", "BVAL"]

# Textural features
TexturalFeature = ["contrast", "energy", "homogeneity", "correlation", "entropy"]

# ROI size (half)
ROISize_T2 = 20
ROISize_ADC = 10

# source locations
DicomPath = "../DATA/ProstateEX/Rawdata/TEST/DOI"
CSVPath = "../DATA/ProstateEX/csv"
CSVFindings = "ProstateX-2-Findings-Test.csv"
CSVImages = "ProstateX-2-Images-Test.csv"

# ROI image locations
ROIImagePath = "../DATA/ProstateEX/Preprocessing/TEST/2D_PNG"
ROINumpyPath = "../DATA/ProstateEX/Preprocessing/TEST/2D"

print("preprocessing start")
start_time = time.time()

# open csv file
with open(os.path.join(CSVPath, CSVImages), 'r', newline='') as CSV_images_object:
    CSV_images_reader = csv.DictReader(CSV_images_object)

    for row in CSV_images_reader:
        '''
        # Select image sequence
        selected_sequence = ImageSequence[0]

        if row['DCMSerDescr'] == selected_sequence:
        '''

        # get sequence name
        selected_sequence = row['DCMSerDescr']

        # set lesion location
        lesion_location = row['ijk'].split()

        # set image path
        patient_root_path = os.path.join(DicomPath, row['ProxID'])
        for images in os.listdir(patient_root_path):
            image_path = os.path.join(patient_root_path, images, row['DCMSerUID'])
            print(row['ProxID'], row['DCMSerDescr'])

            # Read dicom files
            onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
            for file in onlyfiles:
                ds = dicom.read_file(os.path.join(image_path, file))

                # set ROI size
                if selected_sequence == ImageSequence[0] or selected_sequence == ImageSequence[1]:
                    ROISize = ROISize_T2
                else:
                    ROISize = ROISize_ADC

                # calculate Z axis extent
                image_dimension = row['Dim'].split('x')
                slice_num = int(image_dimension[2]) - int(ds[0x20, 0x13].value)
                voxel_spacing = row['VoxelSpacing'].split(',')
                # ROISlice = round((ROISize*float(voxel_spacing[0]))/float(voxel_spacing[2])/2, 0)
                ROISlice = 2

                # get ROI image
                if abs(int(slice_num) - int(lesion_location[2])) <= ROISlice:
                    ROI_i0 = int(lesion_location[0]) - ROISize
                    ROI_i1 = int(lesion_location[0]) + ROISize
                    ROI_j0 = int(lesion_location[1]) - ROISize
                    ROI_j1 = int(lesion_location[1]) + ROISize

                    ROI_image = ds.pixel_array[ROI_j0:ROI_j1, ROI_i0:ROI_i1]

                    # save ROI image
                    # as PNG
                    image_filename = row['ProxID'] + "_" + row['fid'] + "_" + selected_sequence + "_" + str(slice_num)
                    scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'ORI', image_filename) + ".png", ROI_image)

                    # as npy
                    np.save(os.path.join(ROINumpyPath, selected_sequence, 'ORI', image_filename), ROI_image)

                    #
                    # make textural feature map
                    #

                    # calculate textural feature for every voxel from glcm using 5x5 patch centered at each voxel
                    feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
                    y = 0
                    for j in range(ROI_j0, ROI_j1):
                        x = 0
                        for i in range(ROI_i0, ROI_i1):
                            ROI_image = ds.pixel_array[j-2:j+2, i-2:i+2]

                            ROI_image_uint8 = np.zeros(shape=ROI_image.shape, dtype=np.uint8)
                            ROI_image_uint8 = cv2.convertScaleAbs(ROI_image, alpha=(255.0/65535.0))
                            glcm = skf.greycomatrix(ROI_image_uint8, [1], [0], symmetric=True, normed=True)

                            # make feature map
                            idx = 0
                            for feature in TexturalFeature:
                                if feature == "entropy":
                                    feature_map[y, x, idx] = skm.shannon_entropy(glcm)

                                else:
                                    feature_map[y, x, idx] = skf.greycoprops(glcm, feature)
                                idx += 1

                            x += 1
                        y += 1

                    # save feature map image
                    idx = 0
                    for feature in TexturalFeature:
                        # as PNG
                        image_filename = row['ProxID'] + "_" + row['fid'] + "_" + selected_sequence + "_" + feature + "_" + str(slice_num)
                        scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'TF', image_filename) + ".png", feature_map[:, :, idx])

                        # as npy
                        np.save(os.path.join(ROINumpyPath, selected_sequence, 'TF', image_filename), feature_map[:, :, idx])
                        idx += 1

print("preprocessing end: %s seconds" % (time.time() - start_time))
