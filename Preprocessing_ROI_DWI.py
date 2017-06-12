import dicom
import os
import csv
import time
import scipy.misc
import numpy as np
from read_roi import read_roi_file
from PIL import Image, ImageDraw
import skimage.feature as skf
import skimage.measure as skm
import cv2
from matplotlib import pyplot as plt

# Image Sequence
ImageSequence = ["T2tra", "T2sag", "ADC", "BVAL"]
ImageSequence_ROI = ["T2ax", "T2sag", "ADC", "DWI"]

# Textural features
TexturalFeature = ["contrast", "energy", "homogeneity", "correlation", "entropy"]

# ROI size
ROISize_T2 = (40, 40)
ROISize_ADC = (20, 20)

# source locations
DicomPath = "../DATA/ProstateEX/Rawdata/TRAIN/DOI"
ROIPath = "../DATA/ProstateEX/ROI/TRAIN"
CSVPath = "../DATA/ProstateEX/csv"
CSVFindings = "ProstateX-2-Findings-Train.csv"
CSVImages = "ProstateX-2-Images-Train.csv"

# ROI image locations
ROIImagePath = "../DATA/ProstateEX/Preprocessing/TRAIN_resampled/ROI_PNG"
ROINumpyPath = "../DATA/ProstateEX/Preprocessing/TRAIN_resampled/ROI"

print("preprocessing start")
start_time = time.time()

# read csv file
with open(os.path.join(CSVPath, CSVImages), 'r', newline='') as CSV_images_object:
    for roi_file in os.listdir(ROIPath):
        # parse roi filename
        roi_filename = os.path.splitext(roi_file)[0]
        roi_filename_split = roi_filename.split('_')

        patient = roi_filename_split[0]
        if len(roi_filename_split) == 4:
            fid = int(roi_filename_split[1])
        else:
            fid = 0
        sequence = roi_filename_split[len(roi_filename_split)-2]
        slice = roi_filename_split[len(roi_filename_split)-1]

        print(patient, fid, sequence, slice)

        # read roi file
        roi_object = read_roi_file(os.path.join(ROIPath, roi_file))
        roi_temp = roi_object[roi_filename]

        # find image dimension and path
        CSV_images_object.seek(0)
        CSV_images_reader = csv.DictReader(CSV_images_object)
        for row in CSV_images_reader:
            if row['ProxID'].split('-')[1] == patient:
                if ((len(roi_filename_split) == 4) and (row['fid'] == str(fid))) or (len(roi_filename_split) == 3):
                    if sequence in [ImageSequence_ROI[2], ImageSequence_ROI[3]]:
                        if row['DCMSerDescr'] == ImageSequence[2]:
                            image_dimension = row['Dim'].split('x')
                            image_path = row['DCMSerUID']
                        elif row['DCMSerDescr'] == ImageSequence[3]:
                            image_dimension = row['Dim'].split('x')
                            image_path_DWI = row['DCMSerUID']
                    else:
                        if ((sequence == ImageSequence_ROI[0]) and (row['DCMSerDescr'] == ImageSequence[0]))\
                                or ((sequence == ImageSequence_ROI[1]) and (row['DCMSerDescr'] == ImageSequence[1])):
                            image_dimension = row['Dim'].split('x')
                            image_path = row['DCMSerUID']

        # make polygon
        polygon = []
        for x, y in zip(roi_temp['x'], roi_temp['y']):
            polygon.append(x)
            polygon.append(y)

        # make mask
        width = int(image_dimension[0])
        height = int(image_dimension[1])
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)

        # set image path
        patient_root = "ProstateX-" + patient
        patient_root_path = os.path.join(DicomPath, patient_root)

        for images in os.listdir(patient_root_path):
            if sequence in [ImageSequence_ROI[2], ImageSequence_ROI[3]]:
                image_path = os.path.join(patient_root_path, images, image_path_DWI)

                # Read dicom files
                onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
                for file in onlyfiles:
                    ds = dicom.read_file(os.path.join(image_path, file))

                    # get ROI image
                    slice_num = int(ds[0x20, 0x13].value)
                    if slice_num == int(slice):
                        ROI_i0 = min(roi_temp['x'])
                        ROI_i1 = max(roi_temp['x'])
                        ROI_j0 = min(roi_temp['y'])
                        ROI_j1 = max(roi_temp['y'])

                        # apply mask
                        masked_array = ds.pixel_array * mask
                        ROI_image = masked_array[ROI_j0:ROI_j1, ROI_i0:ROI_i1]

                        # resize ROi image
                        if sequence in [ImageSequence_ROI[0], ImageSequence_ROI[1]]:
                            dim = ROISize_T2
                        else:
                            dim = ROISize_ADC

                            # set sequence = DWI
                            sequence = ImageSequence_ROI[3]
                        resized_ROI_image = cv2.resize(ROI_image, dim, interpolation=cv2.INTER_CUBIC)

                        # display
                        # plt.imshow(resized_ROI_image, cmap='gray')
                        # plt.show()

                        # save ROI image
                        # as PNG
                        image_filename = patient_root + "_" + str(fid) + "_" + str(sequence) + "_" + str(slice_num)
                        scipy.misc.imsave(os.path.join(ROIImagePath, sequence, 'ORI', image_filename) + ".png", resized_ROI_image)

                        # as npy
                        np.save(os.path.join(ROINumpyPath, sequence, 'ORI', image_filename), resized_ROI_image)

                        #
                        # make textural feature map
                        #

                        # calculate textural feature for every voxel from glcm using 5x5 patch centered at each voxel
                        feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
                        y = 0
                        for j in range(ROI_j0, ROI_j1):
                            x = 0
                            for i in range(ROI_i0, ROI_i1):
                                ROI_image_texture = ds.pixel_array[j-2:j+2, i-2:i+2]

                                ROI_image_uint8 = np.zeros(shape=ROI_image_texture.shape, dtype=np.uint8)
                                ROI_image_uint8 = cv2.convertScaleAbs(ROI_image_texture, alpha=(255.0/65535.0))
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

                        # apply mask and resize
                        masked_feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
                        resized_feature_map = np.zeros(shape=(resized_ROI_image.shape[0], resized_ROI_image.shape[1], len(TexturalFeature)))
                        ROI_mask = mask[ROI_j0:ROI_j1, ROI_i0:ROI_i1]
                        idx = 0
                        for feature in TexturalFeature:
                            masked_feature_map[:, :, idx] = feature_map[:, :, idx] * ROI_mask
                            resized_feature_map[:, :, idx] = cv2.resize(masked_feature_map[:, :, idx], dim, interpolation=cv2.INTER_AREA)
                            idx += 1

                        # save feature map image
                        idx = 0
                        for feature in TexturalFeature:
                            # as PNG
                            image_filename = patient_root + "_" + str(fid) + "_" + str(sequence) + "_" + feature + "_" + str(slice_num)
                            scipy.misc.imsave(os.path.join(ROIImagePath, sequence, 'TF', image_filename) + ".png", resized_feature_map[:, :, idx])

                            # as npy
                            np.save(os.path.join(ROINumpyPath, sequence, 'TF', image_filename), resized_feature_map[:, :, idx])
                            idx += 1

print("preprocessing end: %s seconds" % (time.time() - start_time))
