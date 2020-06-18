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

sys.path.append(os.path.expanduser('./Util'))
sys.path.append(os.path.expanduser('./ShapeNet'))

import DataIO_ShapeNet as IO
from ShapeNet import ShapeNet_DGCNN_trainer as trainer
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
parser.add_argument('--GPU','-gpu',type=int,help='GPU to use [default:0]',default=0)
parser.add_argument('--batchsize',type=int,help='Training batchsize [default: 1]',default=1)
parser.add_argument('--m','-m',type=float,help='the ratio of points selected to be labelled [default: 0.01]',default=0.1)
parser.add_argument('--Style','-style',type=str,help='Style for evaluating the network [default: Full]'
                                                     ' [options: Plain, Full]',default='Full')
parser.add_argument('--Network','-net',type=str,help='Network used for training the network [default: DGCNN]'
                                                     ' [options: DGCNN, PointNet++(not supported yet)]',default='DGCNN')
args = parser.parse_args()

##### Set specified GPU to be active
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

##### Load Training/Testing Data
Loader = IO.ShapeNetIO('./Dataset/ShapeNet',batchsize = args.batchsize)
Loader.LoadTestFiles()

##### Evaluation Object
Eval = Evaluation.Eval()

## Number of categories
PartNum = Loader.NUM_PART_CATS
output_dim = PartNum
ShapeCatNum = Loader.NUM_CATEGORIES

#### Save Directories
dt='2020-05-23_10-45-25'
BASE_PATH = os.path.expanduser('./Results/ShapeNet/{}_sty-{}_m-{}_{}'.format(args.Network, args.Style, args.m, dt))
SUMMARY_PATH = os.path.join(BASE_PATH, 'Summary')
PRED_PATH = os.path.join(BASE_PATH, 'Prediction')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'Checkpoint')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')

##### Initialize Inference Operations
TrainOp = trainer.ShapeNet_Trainer()
TrainOp.SetLearningRate(BatchSize=args.batchsize)

##### Define Network
TrainOp.defineNetwork(batch_size=args.batchsize, point_num=3000, style='Plain')

##### Define Label Propagation Solver
if args.Style == 'Full':
    TrainOp.defLabelPropSolver()

##### Restore Checkpoint
best_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{}'.format('best'))
TrainOp.RestoreCheckPoint(best_filepath)

##### Start Testing
printout('\n\nstart Inference at {}\n'.format(time.ctime()))

#### Evaluate
avg_loss, avg_acc, perdata_miou, pershape_miou = TrainOp.Test(Loader,Eval,args.Style)

print('\nAvg Loss {:.4f}  Avg Acc {:.3f}%  Avg PerData IoU {:.3f}%  Avg PerCat IoU {:.3f}%'.format(
    avg_loss, 100 * avg_acc, 100 * np.mean(perdata_miou), 100 * np.mean(pershape_miou)), end='')

string = '\nEval PerShape IoU:'
for iou in pershape_miou:
    string += ' {:.2f}%'.format(100 * iou)
print(string)
