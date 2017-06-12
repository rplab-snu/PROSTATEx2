import dicom
import os
import csv
import time
import scipy.misc
import numpy as np
import SimpleITK as sitk
from read_roi import read_roi_file
from PIL import Image, ImageDraw
import skimage.feature as skf
import skimage.measure as skm
import cv2
from matplotlib import pyplot as plt

# Image Sequence
selected_sequence = "DCE"

# Textural features
TexturalFeature = ["contrast", "energy", "homogeneity", "correlation", "entropy"]

# ROI size
ROISize_DCE = (20, 20)

# source locations
DicomPath = "../DATA/ProstateEX/Rawdata/TRAIN/KtransTrain"
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
with open(os.path.join(CSVPath, CSVFindings), 'r', newline='') as CSV_images_object:
    for roi_file in os.listdir(ROIPath):
        # parse roi filename
        roi_filename = os.path.splitext(roi_file)[0]
        roi_filename_split = roi_filename.split('_')

        patient = roi_filename_split[0]
        sequence = roi_filename_split[len(roi_filename_split)-2]
        if not(sequence == 'kt'):
            continue
        slice = roi_filename_split[len(roi_filename_split)-1]
        if len(roi_filename_split) == 4:
            fid = int(roi_filename_split[1])

            CSV_images_object.seek(0)
            CSV_images_reader = csv.DictReader(CSV_images_object)
            for row in CSV_images_reader:
                if (row['ProxID'].split('-')[1] == patient) and (int(row['fid']) == fid):
                    scanner_coordinate = row['pos'].split()
                    break
        else:
            CSV_images_object.seek(0)
            CSV_images_reader = csv.DictReader(CSV_images_object)
            for row in CSV_images_reader:
                if row['ProxID'].split('-')[1] == patient:
                    fid = int(row['fid'])
                    scanner_coordinate = row['pos'].split()
                    break

        print(patient, fid, sequence, scanner_coordinate, slice)

        # set image path
        patient_root = "ProstateX-" + patient
        patient_root_path = os.path.join(DicomPath, patient_root)
        filename = patient_root + "-Ktrans.mhd"

        # read KTrans image
        image = sitk.ReadImage(os.path.join(patient_root_path, filename))

        # get image array (Z, Y, X)
        image_array = sitk.GetArrayFromImage(image)
        # transpose to (X, Y, Z)
        image_array = image_array.transpose(2, 1, 0)
        image_array = np.fliplr(image_array)
        image_array = np.rot90(image_array, 1)

        # get image coordinate (ijk)
        xyz = [float(i) for i in scanner_coordinate]
        ijk = image.TransformPhysicalPointToIndex(xyz)

        # get image shape
        shape = image_array.shape

        # write ijk location on csv file
        # csv_location_writer.writerow([patient, fid, str(ijk)[1:-1]])

        # verbose
        # print("##############" + patient)
        # print("image dimension: ", image_array.shape)
        # print("voxel size: ", image.GetSpacing())
        # print("scanner coordinate: ", scanner_coordinate)
        # print("image coordinate: ", ijk)

        # read roi file
        roi_object = read_roi_file(os.path.join(ROIPath, roi_file))
        roi_temp = roi_object[roi_filename]

        # make polygon
        polygon = []
        for x, y in zip(roi_temp['x'], roi_temp['y']):
            polygon.append(x)
            polygon.append(y)

        # make mask
        width = int(shape[0])
        height = int(shape[1])
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)

        # display
        # plt.imshow(mask, cmap='gray')
        # plt.show()

        ROI_i0 = min(roi_temp['x'])
        ROI_i1 = max(roi_temp['x'])
        ROI_j0 = min(roi_temp['y'])
        ROI_j1 = max(roi_temp['y'])

        # apply mask
        slice_array = image_array[:, :, int(slice)-1]
        masked_array = slice_array * mask
        ROI_image = masked_array[ROI_j0:ROI_j1, ROI_i0:ROI_i1]

        # resize ROi image
        dim = ROISize_DCE
        resized_ROI_image = cv2.resize(ROI_image, dim, interpolation=cv2.INTER_CUBIC)

        # display
        # plt.imshow(slice_array, cmap='gray')
        # plt.show()
        # plt.imshow(masked_array, cmap='gray')
        # plt.show()
        # plt.imshow(resized_ROI_image, cmap='gray')
        # plt.show()

        # save ROI image
        # as PNG
        image_filename = patient_root + "_" + str(fid) + "_" + selected_sequence + "_" + slice
        scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'ORI', image_filename) + ".png", resized_ROI_image)

        # as npy
        np.save(os.path.join(ROINumpyPath, selected_sequence, 'ORI', image_filename), resized_ROI_image)

        #
        # make textural feature map
        #

        # calculate textural feature for every voxel from glcm using 5x5 patch centered at each voxel
        feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
        y = 0
        for j in range(ROI_j0, ROI_j1):
            x = 0
            for i in range(ROI_i0, ROI_i1):
                ROI_image_texture = image_array[j - 2:j + 2, i - 2:i + 2, int(slice)-1]

                ROI_image_uint8 = np.zeros(shape=ROI_image_texture.shape, dtype=np.uint8)
                ROI_image_uint8 = cv2.convertScaleAbs(ROI_image_texture, alpha=(255.0 / 65535.0))
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
            resized_feature_map[:, :, idx] = cv2.resize(masked_feature_map[:, :, idx], dim, interpolation=cv2.INTER_CUBIC)
            idx += 1

        # save feature map image
        idx = 0
        for feature in TexturalFeature:
            # as PNG
            image_filename = patient_root + "_" + str(fid) + "_" + selected_sequence + "_" + feature + "_" + slice
            scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'TF', image_filename) + ".png",
                              resized_feature_map[:, :, idx])

            # as npy
            np.save(os.path.join(ROINumpyPath, selected_sequence, 'TF', image_filename), resized_feature_map[:, :, idx])
            idx += 1

print("preprocessing end: %s seconds" % (time.time() - start_time))
