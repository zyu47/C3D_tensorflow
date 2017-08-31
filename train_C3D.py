
import numpy as np
import os
import random
from os.path import join
import tensorflow as tf

from neural_nets.solver import Solver
from data_preprocessing.c3d_preprocessing import *
from models.C3D import C3D


'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()
'''

logpath = './c3d-ucf/'
data_dir = '/s/red/a/nobackup/vision/UCF-101/'
label_path = './ucf-101-labels.npy'
trainlist_path = '/s/chopin/k/grad/dkpatil/Downloads/UCF-101/ucfTrainTestlist/trainlist01.txt'
testlist_path = '/s/chopin/k/grad/dkpatil/Downloads/UCF-101/ucfTrainTestlist/testlist01.txt'

'''
labels = list(set([i.split('_')[1] for i in video_file_names]))
lab_ind = {}
for ind in range(101):
    lab_ind[labels[ind]] = ind
np.save('ucf-101-labels.npy', lab_ind)
print len(video_file_names), len(labels)

print len(set(labels))
'''


def readList():
    f = open(trainlist_path, 'r')

    lab_ind = {}
    trainlist_x = []
    trainlist_y = []
    for line in f:
        label = int(line.split()[1]) - 1
        video_filename = line.split()[0].split('/')[1]
        lab_ind[line.split()[0].split('/')[0]] = label
        trainlist_x.append(join(data_dir, video_filename))#append(video_filename)
        trainlist_y.append(label)
        #break

    trainlist = zip(trainlist_x, trainlist_y)
    # print(len(trainlist))
    # print(trainlist[0])
    # print(lab_ind)
    f.close()

    f = open(testlist_path, 'r')
    testlist_x = []
    testlist_y = []
    for line in f:
        video_filename = line.rstrip().split('/')[1]
        label = lab_ind[line.rstrip().split('/')[0]]
        testlist_x.append(join(data_dir, video_filename))#append(video_filename)
        testlist_y.append(label)
        #break

    testlist = zip(testlist_x, testlist_y)
    # print(len(testlist))
    # print(testlist[0])
    f.close()
    return np.array(trainlist_x), np.array(trainlist_y), np.array(testlist_x), np.array(testlist_y)
    #trainlist, testlist


def read_all_clips(verbose = False):
    '''
    trainlist, testlist = readList()
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    data = {}

    for (video_file_name, label) in trainlist:
        video_path = join(data_dir, video_file_name)
        processed = proc_video(video_path)
        ytrain += [label for i in range(len(processed))]
        xtrain += processed
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    for (video_file_name, label) in testlist:
        video_path = join(data_dir, video_file_name)
        processed = proc_video(video_path)
        ytest += [label for i in range(len(processed))]
        xtest += processed
    xtest = np.array(xtest)
    ytest = np.array(ytest)

    if verbose:
        print xtrain.shape, xtrain[0].shape
        print ytrain.shape, ytrain
        print xtest.shape, xtest[0].shape
        print ytest.shape, ytest
    '''

    data = {}
    xtrain, ytrain, xval, yval = readList()

    ind_dict_train = {}
    for i in set(ytrain):
        ind_dict_train[i] = []
    ind_dict_val = {}
    for i in set(yval):
        ind_dict_val[i] = []

    for i in range(len(ytrain)):
        ind_dict_train[ytrain[i]].append(i)
    for i in range(len(yval)):
        ind_dict_val[yval[i]].append(i)

    data['X_train'] = xtrain
    data['y_train'] = ytrain
    data['X_val'] = xval
    data['y_val'] = yval
    data['index_dict_train'] = ind_dict_train
    data['index_dict_val'] = ind_dict_val
    return data

#data = read_all_clips()
c3d = C3D(logpath)
solver = Solver(model= c3d, data=read_all_clips(), learning_rate=3e-2, num_epochs=16, batch_size=30, lr_decay=10, dropout=0.25 )
solver.run_validation()
