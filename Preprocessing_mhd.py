import os
import csv
import time
import SimpleITK as sitk
from matplotlib import pyplot as plt
import scipy.misc
import numpy as np
import skimage.feature as skf
import skimage.measure as skm
import cv2

# Image Sequence
selected_sequence = "DCE"

# Textural features
TexturalFeature = ["contrast", "energy", "homogeneity", "correlation", "entropy"]

# ROI size (half)
ROISize = 20

# source locations
DicomPath = "../DATA/ProstateEX/Test/KtransTest"
CSVPath = "../DATA/ProstateEX/csv"
CSVFindings = "ProstateX-2-Findings-Test.csv"
CSVImages = "ProstateX-2-Images-Test.csv"
CSVDCELocation = "ProstateX-2-DCE-Test-Location.csv"

# ROI image locations
ROIImagePath = "../DATA/ProstateEX/Preprocessing/TEST/2D_PNG"
ROINumpyPath = "../DATA/ProstateEX/Preprocessing/TEST/2D"

print("preprocessing start")
start_time = time.time()

# csv for write location (image coordinate)
csv_location_object = open(os.path.join(CSVPath, CSVDCELocation), 'w', newline='')
csv_location_writer = csv.writer(csv_location_object)
csv_location_writer.writerow(["ProxID","fid","ijk"])

# open csv file
with open(os.path.join(CSVPath, CSVFindings), 'r', newline='') as CSV_images_object:
    CSV_images_reader = csv.DictReader(CSV_images_object)

    for row in CSV_images_reader:
        patient = row['ProxID']
        fid = row['fid']

        # read KTrans image
        filename = patient + "-Ktrans.mhd"
        image = sitk.ReadImage(os.path.join(DicomPath, patient, filename))

        # get image array (Z, Y, X)
        image_array = sitk.GetArrayFromImage(image)
        # transpose to (X, Y, Z)
        image_array = image_array.transpose(2, 1, 0)

        # get image coordinate (ijk)
        scanner_coordinate = row['pos'].split()
        xyz = [float(i) for i in scanner_coordinate]
        ijk = image.TransformPhysicalPointToIndex(xyz)

        # write ijk location on csv file
        csv_location_writer.writerow([patient, fid, str(ijk)[1:-1]])

        # verbose
        print("##############" + patient)
        print("image dimension: ", image_array.shape)
        print("voxel size: ", image.GetSpacing())
        print("scanner coordinate: ", scanner_coordinate)
        print("image coordinate: ", ijk)

        # calculate Z axis extent
        shape = image_array.shape
        ROISlice = 2
        for slice in range(0, shape[2]):
            # get ROI image
            slice_num = shape[2] - slice
            if abs(int(slice_num) - int(ijk[2])) <= ROISlice:
                ROI_i0 = int(ijk[0]) - ROISize
                ROI_i1 = int(ijk[0]) + ROISize
                ROI_j0 = int(ijk[1]) - ROISize
                ROI_j1 = int(ijk[1]) + ROISize

                ROI_image = image_array[ROI_i0:ROI_i1, ROI_j0:ROI_j1, slice]

                # save ROI image
                # as PNG
                image_filename = row['ProxID'] + "_" + row['fid'] + "_" + selected_sequence + "_" + str(slice_num)
                scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'ORI', image_filename) + ".png",
                                  ROI_image)

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
                        ROI_image = image_array[i - 2:i + 2, j - 2:j + 2, slice]

                        ROI_image_uint8 = np.zeros(shape=ROI_image.shape, dtype=np.uint8)
                        # ROI_image_uint8 = cv2.convertScaleAbs(ROI_image, alpha=(255.0 / 65535.0))
                        ROI_image_uint8 = cv2.convertScaleAbs(ROI_image)
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
                    image_filename = patient + "_" + fid + "_" + selected_sequence + "_" + feature + "_" + str(slice_num)
                    scipy.misc.imsave(os.path.join(ROIImagePath, selected_sequence, 'TF', image_filename) + ".png",
                                      feature_map[:, :, idx])

                    # as npy
                    np.save(os.path.join(ROINumpyPath, selected_sequence, 'TF', image_filename), feature_map[:, :, idx])
                    idx += 1

            # plt.imshow(image_array[:, :, slice], cmap='gray')
            # plt.show()

csv_location_object.close()
print("preprocessing end: %s seconds" % (time.time() - start_time))
