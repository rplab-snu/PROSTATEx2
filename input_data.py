import sugartensor as sg
import numpy as np
import csv
import tensorflow as tf
import os

@sg.sg_producer_func
def _load_data(src_list):
    labels, data = src_list

    labels = np.array(labels)
    data = np.array(data, np.float)

    return labels, data


def allfiles(path):    # file 명 가져오기
    list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            list.append(file)

    return list


patient_num_fid_GGG = []
with open('ProstateX-2-Findings-Train_edit.csv') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        patient_num_fid_GGG.append(row[0:3])


def find_GGG(filename):
    patient_number = filename[:14]
    fid = filename[15]
    candidate_1 = [s for s in patient_num_fid_GGG if patient_number in s]
    candidate_2 = []
    for i in candidate_1:
        if fid == i[1]:
            candidate_2.append(i)
    GGG = candidate_2[0][2]

    return int(GGG) - 1


class Surv(object):

    def __init__(self, batch_size=32):
        ORI_list = allfiles('ORI')

        data, labels = [], []

        for i in ORI_list:
            img = np.load('ORI/' + i)
            data.append(img)                 # (40 x 40)
            G = find_GGG(i)
            labels.append(G)

        print(type(labels[0]))

        label_t = tf.convert_to_tensor(labels)
        data_t = tf.convert_to_tensor(np.array(data))

        label_q, data_q\
            = tf.train.slice_input_producer([label_t, data_t], shuffle=True)

        label_q, data_q = _load_data(source=[label_q, data_q],
                                     dtypes=[sg.sg_intx, sg.sg_floatx],
                                     capacity=256, num_threads=64)

        batch_queue = tf.train.batch([label_q, data_q], batch_size=batch_size,
                                     shapes=[(), (40, 40)],
                                     num_threads=512, capacity=batch_size*32,
                                     dynamic_pad=False)

        self.label, self.data = batch_queue

        print(self.data.get_shape())
        self.data = tf.reshape(self.data, [-1, 40, 40, 1])
        self.num_batch = len(labels) // 32

        sg.sg_info('%s set loaded.(total data=%d, total batch=%d)'
                   % ('train', len(labels), self.num_batch))



