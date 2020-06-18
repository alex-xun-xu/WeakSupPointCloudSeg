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
sys.path.append(os.path.expanduser('./ShapeNet'))

import DataIO_ShapeNet as IO
from ShapeNet import ShapeNet_DGCNN_trainer as trainer
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
parser.add_argument('--batchsize','-bs',type=int,help='Training batchsize [default: 1]',default=6)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.1)
parser.add_argument('--Style','-sty',type=str,help='Style for training the network [default: Full]'
                                                     ' [options: Plain, Full]',default='Full')
parser.add_argument('--Network','-net',type=str,help='Network used for training the network [default: DGCNN]'
                                                     ' [options: DGCNN, PointNet++(not supported yet)]',default='DGCNN')
args = parser.parse_args()

##### Set specified GPU to be active
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

##### Load Training/Testing Data
Loader = IO.ShapeNetIO('./Dataset/ShapeNet',batchsize = args.batchsize)
Loader.LoadTrainValFiles()

##### Evaluation Object
Eval = Evaluation.ShapeNetEval()

## Number of categories
PartNum = Loader.NUM_PART_CATS
output_dim = PartNum
ShapeCatNum = Loader.NUM_CATEGORIES

#### Export results Directories
if args.ExpRslt:
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get current time
    BASE_PATH = os.path.expanduser('./Results/ShapeNet/{}_sty-{}_m-{}_{}'.format(args.Network, args.Style, args.m, dt))
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

##### Initialize Training Operations
TrainOp = trainer.ShapeNet_Trainer()
TrainOp.SetLearningRate(LearningRate=args.LearningRate,BatchSize=args.batchsize)

##### Define Network
TrainOp.defineNetwork(batch_size=args.batchsize, point_num=2048, style=args.Style, rampup=args.Rampup)

#### Load Sampled Point Index
save_path = os.path.expanduser('./Dataset/ShapeNet/Preprocess/')
save_filepath = os.path.join(save_path, 'SampIndex_m-{:.3f}.mat'.format(args.m))
tmp = scio.loadmat(save_filepath)
pts_idx_list = tmp['pts_idx_list']
file_idx_list = np.zeros(shape=[pts_idx_list.shape[0]])
data_idx_list = np.arange(0,pts_idx_list.shape[0])

##### Start Training Epochs
for epoch in range(0,args.Epoch):

    if args.ExpRslt:
        fid = open(summary_filepath, 'a')
    else:
        fid = None

    printout('\n\nstart {:d}-th epoch at {}\n'.format(epoch, time.ctime()),write_flag = args.ExpRslt, fid=fid)

    #### Shuffle Training Data
    Loader.Shuffle_TrainSet()

    #### Train One Epoch
    train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch(Loader,file_idx_list,data_idx_list,pts_idx_list)

    printout('\nTrainingSet  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
        train_avg_loss, 100 * train_avg_acc), write_flag = args.ExpRslt, fid = fid)

    #### Evaluate on Validation Set One Epoch
    if epoch % 5 ==0:
        eval_avg_loss, eval_avg_acc, eval_perdata_miou, eval_pershape_miou = TrainOp.EvalOneEpoch(Loader, Eval)

        printout('\nEvaluationSet   avg loss {:.2f}   acc {:.2f}%   PerData mIoU {:.3f}%   PerShape mIoU {:.3f}%'.
              format(eval_avg_loss,100*eval_avg_acc, 100*np.mean(eval_perdata_miou), 100*np.mean(eval_pershape_miou)), write_flag = args.ExpRslt, fid = fid)

        string = '\nEval PerShape IoU:'
        for iou in eval_pershape_miou:
            string += ' {:.2f}%'.format(100 * iou)
        printout(string, write_flag = args.ExpRslt, fid = fid)

        if args.ExpRslt:
            save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{:d}'.format(epoch))
            best_filename = 'Checkpoint_epoch-{}'.format('best')

            TrainOp.SaveCheckPoint(save_filepath, best_filename, np.mean(eval_pershape_miou))

    if args.ExpRslt:
        fid.close()


