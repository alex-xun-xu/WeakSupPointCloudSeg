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

sys.path.append(os.path.abspath('./Util'))
sys.path.append(os.path.abspath('./S3DIS'))

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
                                                     ' [options: Plain, Full]',default='Plain')
parser.add_argument('--Network','-net',type=str,help='Network used for training the network [default: DGCNN]'
                                                     ' [options: DGCNN, PointNet++(not supported yet)]',default='DGCNN')
args = parser.parse_args()

##### Set specified GPU to be active
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)


##### Load Training/Testing Data
PartNum = 13
output_dim = PartNum
num_points = 4096
Loader = IO.S3DIS_IO('./Dataset/S3DIS/indoor3d_sem_seg_hdf5_data', PartNum, batchsize=args.batchsize)
Loader.LoadS3DIS_AllData()
Loader.CreateDataSplit(args.test_area)
Loader.ResetLoader_TrainSet()

##### Evaluation Object
Eval = Evaluation.Eval()

##### Initialize Training Operations
TrainOp = trainer.S3DIS_Trainer(args.test_area)
TrainOp.SetLearningRate(LearningRate=args.LearningRate,BatchSize=args.batchsize)

#### Export results Directories
if args.ExpRslt:
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get current time
    BASE_PATH = os.path.expanduser('./Results/S3DIS/{}_sty-{}_m-{}_{}'.format(args.Network, args.Style, args.m, dt))
    SUMMARY_PATH = os.path.join(BASE_PATH,'Summary'.format(args.m))
    PRED_PATH = os.path.join(BASE_PATH,'Prediction'.format(args.m))
    CHECKPOINT_PATH = os.path.join(BASE_PATH,'Checkpoint'.format(args.m))

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    if not os.path.exists(SUMMARY_PATH):
        os.makedirs(SUMMARY_PATH)

    if not os.path.exists(PRED_PATH):
        os.makedirs(PRED_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')
    fid = open(summary_filepath,'w')
    fid.close()

##### Define Network
if args.Style == 'Full':
    TrainOp.defineNetwork(batch_size=2*args.batchsize, num_points=num_points, style=args.Style, rampup=args.Rampup)
elif args.Style == 'Plain':
    TrainOp.defineNetwork(batch_size=args.batchsize, num_points=num_points, style=args.Style, rampup=args.Rampup)

#### Load Sampled Point Index
save_path = os.path.expanduser('./Dataset/S3DIS/Preprocess/')
save_filepath = os.path.join(save_path,'SampIndex_m-{:.3f}.mat'.format(args.m))
tmp = scio.loadmat(save_filepath)
if args.m == 0:
    pts_idx_list = []
    for b_i in range(tmp['pts_idx_list'].shape[1]):
        pts_idx_list.append(tmp['pts_idx_list'][0,b_i][0])
else:
    pts_idx_list = tmp['pts_idx_list']

##### Start Training Epochs
for epoch in range(0,args.Epoch):

    if args.ExpRslt:
        fid = open(summary_filepath, 'a')
    else:
        fid = None

    printout('\n\nstart training {:d}-th epoch at {}\n'.format(epoch, time.ctime()),write_flag = args.ExpRslt, fid=fid)

    #### Shuffle Training Data
    Loader.Shuffle_TrainSet()

    #### Train One Epoch
    if args.Style == 'Full':
        train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch_Full(Loader, pts_idx_list, args.batchsize)
    elif args.Style == 'Plain':
        train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch(Loader,pts_idx_list, args.batchsize)

    printout('\nTrainingSet  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
        train_avg_loss, 100 * train_avg_acc), write_flag = args.ExpRslt, fid = fid)

    #### Evaluate on Validation Set One Epoch
    if epoch % 5 ==0:

        printout('\n\nstart validation {:d}-th epoch at {}\n'.format(epoch, time.ctime()), write_flag=args.ExpRslt, fid=fid)

        if args.Style == 'Full':
            eval_avg_loss, eval_avg_acc, eval_miou = TrainOp.EvalOneEpoch_Full(Loader)
        elif args.Style == 'Plain':
            eval_avg_loss, eval_avg_acc, eval_miou = TrainOp.EvalOneEpoch(Loader)

        printout('\nEvaluationSet   avg loss {:.2f}   acc {:.2f}%   PerData mIoU {:.3f}%'.
                 format(eval_avg_loss, 100 * eval_avg_acc, 100 * np.mean(eval_miou)), write_flag=args.ExpRslt, fid=fid)


        if args.ExpRslt:
            save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{:d}'.format(epoch))
            best_filename = 'Checkpoint_epoch-{}'.format('best')

            TrainOp.SaveCheckPoint(save_filepath, best_filename, np.mean(eval_miou))

    if args.ExpRslt:
        fid.close()


