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
import time
import scipy.io as scio
import datetime

sys.path.append(os.path.expanduser('./Util'))
sys.path.append(os.path.expanduser('./S3DIS'))

import DataIO_S3DIS as IO
from S3DIS import S3DIS_DGCNN_trainer as trainer
from Tool import printout
import Evaluation

##### Take input arguments
parser = arg.ArgumentParser(description='Take parameters')
parser.add_argument('--GPU','-gpu',type=int,help='GPU to use [default:0]',default=0)
parser.add_argument('--ExpRslt','-er',type=bool,help='Flag to indicate if export results [default:False]',default=False)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-3)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train [default: 101]',default=201)
parser.add_argument('--Rampup','-rp',type=int,help='Number of epochs to rampup the weakly supervised losses [default: 51]',default=101)
parser.add_argument('--batchsize','-bs',type=int,help='Training batchsize [default: 1]',default=3)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.1)
parser.add_argument('--test_area','-ta',type=int,help='Test area 1 to 6 [default: 6]',default=5)
parser.add_argument('--Style','-sty',type=str,help='Style for training the network [default: Full]'
                                                     ' [options: Plain, Full]',default='Full')
parser.add_argument('--Network','-net',type=str,help='Network used for training the network [default: DGCNN]'
                                                     ' [options: DGCNN, PointNet++(not supported yet)]',default='DGCNN')
args = parser.parse_args()

##### Set specified GPU to be active
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

## Parameters
Model = 'DGCNN_IncompleteSup_Siamese_MIL_SmoothAdd_v1'
PartNum = 13
output_dim = PartNum
num_points = 4096

#### Save Directories
dt='2020-06-17_16-38-41'
BASE_PATH = os.path.expanduser('./Results/S3DIS/{}_sty-{}_m-{}_{}'.format(args.Network, args.Style, args.m, dt))
SUMMARY_PATH = os.path.join(BASE_PATH, 'Summary')
PRED_PATH = os.path.join(BASE_PATH, 'Prediction')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'Checkpoint')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')

##### Load Testing Data
Loader = IO.S3DIS_Test('area{}'.format(args.test_area),)

##### Evaluation Object
Eval = Evaluation.Eval()

##### Initialize Training Operations
TrainOp = trainer.S3DIS_Trainer(args.test_area)
TrainOp.SetLearningRate(LearningRate=args.LearningRate,BatchSize=args.batchsize)

##### Define Network
# if args.Style == 'Full':
TrainOp.defineNetwork(batch_size=1, num_points=num_points, style=args.Style, rampup=args.Rampup)
# elif args.Style == 'Plain':
#     TrainOp.defineNetwork(batch_size=args.batchsize, num_points=num_points, style=args.Style, rampup=args.Rampup)

##### Define Label Propagation Solver
if args.Style == 'Full':
    TrainOp.defLabelPropSolver()

# ##### Define Label Propagation Solver
# LPSolver = PLP.LabelPropagation_Baseline_TF(alpha = 1e0, beta = 1e0, K=13)
# ## Define TF Computation Operations
# TFComp = {}
# # TFComp['InnerProd'] = Tool.TF_Computation.InnerProd()
# # TFComp['PairDist2'] = Tool.TF_Computation.PairDist2()
# # TFComp['LaplacianMat'] = Tool.TF_Computation.LaplacianMat()
# # TFComp['LaplacianMatSym'] = Tool.TF_Computation.LaplacianMatSym()
# # TFComp['LaplacianMatSym_DirectComp'] = Tool.TF_Computation.LaplacianMatSym_DirectComp()
# TFComp['Lmat'] = Tool.TF_Computation.LaplacianMatSym_XYZRGB_DirectComp()
# # TFComp['Lmat'] = Tool.TF_Computation.LaplacianMat_XYZRGB_DirectComp()

##### Restore Checkpoint
# save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{}'.format('best'))
save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{}'.format('best'))
TrainOp.RestoreCheckPoint(save_filepath)

##### Test
true_positive_classes, positive_classes, gt_classes = TrainOp.Test(Loader,PRED_PATH)

iou = 100*true_positive_classes /( gt_classes + positive_classes - true_positive_classes + 1e-5)
print('Per Category IoU:')
for i in iou:
    print('{:.3f}'.format(i),end=' ')

##### Save Prediction Results
save_filepath = os.path.join(RSLT_PATH,'testarea-{:d}_m-{}_TeLP.mat'.format(args.test_area,args.m))
scio.savemat(save_filepath,{'true_positive_classes':true_positive_classes,'positive_classes':positive_classes,'gt_classes':gt_classes})
