from C3D import C3D
from solver import Solver
from c3d_preprocessing import *


import numpy as np
import os
import random
from os.path import join
import tensorflow as tf



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logpath = './c3d-ucf/'
data_dir = '/s/red/a/nobackup/vision/UCF-101/'
label_path = './ucf-101-labels.npy'
trainlist_path = '/s/chopin/k/grad/zhixian/DraperLab/C3D/ucfTrainTestlist/trainlist01.txt'
testlist_path = '/s/chopin/k/grad/zhixian/DraperLab/C3D/ucfTrainTestlist/testlist01.txt'
c3d_feature_dir = '/s/red/a/nobackup/vision/jason/C3D/'

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

def test_clips(xtest,ytest, testFunction = False, verbose = False):
	
	for ind, vid_path in enumerate(xtest):
		processed = proc_video(vid_path, return_all=True, test_flag=True, verbose= verbose)
		print 'Test file #%d, path: %s, # of clips: %d' %(ind, vid_path, len(processed))
		yield (vid_path.split('/')[-1], processed, [ytest[ind] for i in range(len(processed))])
		if testFunction and ind == 0:
			break

xtrain, ytrain, xtest, ytest = readList()
fname, clips, y = test_clips(xtest, ytest, True, True)
print(fname)
print(clips.shape)
print(y)
#c3d = C3D(logpath)
#c3d.saver.restore(c3d.sess, c3d.logs_path)


#for it in range(solver.y_val_data.shape[0] // solver.batch_size):
	#x_valid_data = solver.X_val_data[(it * solver.batch_size): (it + 1) * solver.batch_size, :, :, :, :]
	##yvalid_lbl = solver.y_val_data[(it * solver.batch_size): (it + 1) * solver.batch_size]
	#summ, acc, ls = solver.model.sess.run(solver.model.h_fc1,
								  #feed_dict={solver.model.x: x_valid_data})#, solver.model.y_: yvalid_lbl, solver.model.keep_prob: 1.0})
	#accuracies.append(acc)
	#losses.append(ls)

