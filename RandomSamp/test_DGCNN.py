import os
import sys
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse as arg
import json
# import deepdish
import time
import scipy.io as scio
from datetime import datetime

sys.path.append(os.path.expanduser('../Util'))
sys.path.append(os.path.expanduser('../ShapeNet'))

import DataIO_ShapeNet as IO
from ShapeNet import ShapeNet_DGCNN_util as util
import Tool
from Tool import printout
import Evaluation

def pc_normalize(pc):
    ## Normalize the point cloud by the largest distance to origin
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--ExpSum','-es',type=int,help='Flag to indicate if export summary',default=0)    # bool
parser.add_argument('--SaveMdl','-sm',type=int,help='Flag to indicate if save learned model',default=0)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-4)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train',default=250)
parser.add_argument('--gamma',type=float,help='L2 regularization coefficient',default=0.1)
parser.add_argument('--batchsize',type=int,help='Training batchsize [default: 1]',default=1)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.01)
args = parser.parse_args()

if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

#### Parameters
Model = 'DGCNN_RandomSamp'
# m = 1    # the ratio of points selected to be labelled

##### Load Training/Testing Data
# Loader = IO.ShapeNetIO('../Dataset/ShapeNet',batchsize = args.batchsize)
Loader = IO.ShapeNetIO('/home/xuxun/Dropbox/Dataset/ShapeNet',batchsize = args.batchsize)
Loader.LoadTestFiles()


##### Evaluation Object
Eval = Evaluation.ShapeNetEval()

## Number of categories
PartNum = Loader.NUM_PART_CATS
output_dim = PartNum
ShapeCatNum = Loader.NUM_CATEGORIES

#### Save Directories
dt = str(datetime.now().strftime("%Y%m%d"))
BASE_PATH = os.path.expanduser('../Results/{}/ShapeNet/'.format(dt))
SUMMARY_PATH = os.path.join(BASE_PATH,Model,'Summary_m-{:.3f}'.format(args.m))
CAM_PATH = os.path.join(BASE_PATH,Model,'CAM_m-{:.3f}'.format(args.m))
PRED_PATH = os.path.join(BASE_PATH,Model,'Prediction_m-{:.3f}'.format(args.m))
CHECKPOINT_PATH = os.path.join(BASE_PATH,Model,'Checkpoint_m-{:.3f}'.format(args.m))

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)

if not os.path.exists(CAM_PATH):
    os.makedirs(CAM_PATH)

if not os.path.exists(PRED_PATH):
    os.makedirs(PRED_PATH)

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')


##### Initialize Training Operations
TrainOp = util.ShapeNet_IncompleteSup()
TrainOp.SetLearningRate(LearningRate=args.LearningRate,BatchSize=args.batchsize)

##### Define Network
TrainOp.DGCNN_SemiSup(batch_size=args.batchsize, point_num=3000)

##### Restore Checkpoint
best_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{}'.format('best'))
TrainOp.RestoreCheckPoint(best_filepath)

##### Start Testing
printout('\n\nstart Inference at {}\n'.format(time.ctime()))

#### Evaluate
avg_loss, avg_acc, perdata_miou, pershape_miou = TrainOp.Test(Loader,Eval)

print('\nAvg Loss {:.4f}  Avg Acc {:.3f}%  Avg PerData IoU {:.3f}%  Avg PerCat IoU {:.3f}%'.format(
    avg_loss, 100 * avg_acc, 100 * np.mean(perdata_miou), 100 * np.mean(pershape_miou)), end='')

string = '\nEval PerShape IoU:'
for iou in pershape_miou:
    string += ' {:.2f}%'.format(100 * iou)
print(string)
