import numpy as np
import tensorflow as tf
import os
import time
import Loss
import importlib
import copy
import scipy.sparse as sparse
import sys

sys.path.append(os.path.expanduser('../Util'))

import Tool
import DGCNN_ShapeNet as network
import SmoothConstraint

class ShapeNet_Trainer():

    def __init__(self):

        self.bestValCorrect = 0.


    def SetLearningRate(self,LearningRate,BatchSize):

        self.BASE_LEARNING_RATE = LearningRate
        self.BATCH_SIZE = BatchSize
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.DECAY_STEP = 16881 * 20
        self.DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP*2)
        self.BN_DECAY_CLIP = 0.99

    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(
            self.BASE_LEARNING_RATE,  # Base learning rate.
            self.batch * self.BATCH_SIZE,  # Current index into the dataset.
            self.DECAY_STEP,  # Decay step.
            self.DECAY_RATE,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)  # CLIP THE LEARNING RATE!!
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(
            self.BN_INIT_DECAY,
            self.batch * self.BATCH_SIZE,
            self.BN_DECAY_DECAY_STEP,
            self.BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def defineNetwork(self, batch_size, point_num=2048, loss='Full', rampup=51):
        '''
        define DGCNN network for incomplete labels as supervision
        Args:
            batch_size: batchsize for training network
            point_num: number of points for each point cloud sample
            loss: loss used for training weakly supervised segmentation network
            rampup: rampup epoch for training
        Returns:
        '''

        ##### Define Network Inputs
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num, 3], name='InputPts')  # B*N*3
        self.Y_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size, point_num, 50], name='PartGT')  # B*N*50
        self.Mask_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num], name='Mask')  # B*N
        self.Is_Training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='IsTraining')
        self.Label_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 16],name='ShapeGT') # B*1*K

        ## Set up batch norm decay and learning rate decay
        self.batch = tf.Variable(0, trainable=False)
        bn_decay = self.get_bn_decay()
        learning_rate = self.get_learning_rate()

        ##### Define DGCNN network
        self.Z = network.get_model(self.X_ph, self.Label_ph, self.Is_Training_ph, 16, 50, \
                  batch_size, point_num, 0, bn_decay)

        self.Z_prob = tf.nn.softmax(self.Z, axis=-1)    # posterior output

        ## Segmentation Branch
        loss_seg = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_ph, logits=self.Z)  # B*N
        self.loss_seg = tf.reduce_sum(self.Mask_ph * loss_seg) / tf.reduce_sum(self.Mask_ph)

        ## Final Loss
        self.epoch = tf.Variable(0, trainable=False)
        if loss == 'Plain':
            self.loss = self.loss_seg
        elif loss == 'Full':
            self.WeakSupLoss()
            self.loss = self.loss_seg + \
                        tf.cast(tf.greater_equal(self.epoch,rampup),dtype=tf.float32)*(self.loss_siamese + self.loss_inexact + self.loss_smooth)
        else:
            sys.exit('Loss {} is not defined!'.format(self.loss))

        ##### Define Optimizer
        self.solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=self.batch)  # initialize solver
        self.saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        return True

    def TrainOneEpoch(self, Loader, file_idx_list,data_idx_list,pts_idx_list=None):
        '''
        Function to train one epoch
        :param Loader: Object to load training data
        :param samp_idx_list:  A list indicating the labelled points  B*N
        :return:
        '''
        batch_cnt = 1
        data_cnt = 0
        avg_loss = 0.
        avg_acc = 0.

        while True:

            #### get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet(shuffle_flag = True)

            if not SuccessFlag or batch_cnt > np.inf:
                break

            if mb_size < self.BATCH_SIZE:
                continue

            #### Prepare Incomplete Labelled Training Data
            if pts_idx_list is None:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
            else:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
                for b_i in range(mb_size):
                    batch_samp_idx = \
                        np.where((file_idx_list==file_idx[b_i]) & (data_idx_list==data_idx[b_i]))[0]    # find the data sample from given file and data no.
                    try: Mask_bin[b_i, pts_idx_list[batch_samp_idx][0]] = 1
                    except:
                        continue

            #### Convert Shape Category Label to Onehot Encoding
            label_onehot = Tool.OnehotEncode(label[:,0],Loader.NUM_CATEGORIES)

            #### Remove labels for unlabelled points (to verify)
            seg_onehot_feed = Tool.OnehotEncode(seg,50)

            #### Train One Iteration
            _, loss_mb, Z_prob_mb, batch_no = \
                self.sess.run([self.solver, self.loss, self.Z_prob, self.batch],
                              feed_dict={self.X_ph: data,
                                         self.Y_ph: seg_onehot_feed,
                                         self.Is_Training_ph: True,
                                         self.Mask_ph: Mask_bin,
                                         self.Label_ph:label_onehot})

            ## Calculate loss and correct rate
            avg_loss = (avg_loss * data_cnt + loss_mb*mb_size) / (data_cnt + mb_size)
            pred = []
            for b_i in range(mb_size):
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:,iou_oids] += 1
                pred.append(np.argmax(Z_prob_b,axis=-1))
            pred = np.stack(pred)
            avg_acc = (avg_acc * data_cnt + np.mean(pred==seg)*mb_size) / (data_cnt + mb_size)

            data_cnt += mb_size

            print(
                '\rBatch {:d} TrainedSamp {:d}  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
                    batch_cnt, data_cnt, avg_loss, 100*avg_acc),
                end='')

            batch_cnt += 1

        # increase epoch counter by 1
        self.epoch += 1

        # return avg_loss, perdata_miou, pershape_miou
        return avg_loss, avg_acc

    def EvalOneEpoch(self, Loader, Eval):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)

        while True:

            ## get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_ValSet()

            if not SuccessFlag:
                break

            if mb_size < self.BATCH_SIZE:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [self.BATCH_SIZE - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg, np.tile(seg[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 50)
                label_feed = np.concatenate([label, np.tile(label[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                label_Onehot_feed = Tool.OnehotEncode(label_feed[:, 0], Loader.NUM_CATEGORIES)

            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg, 50)
                label_Onehot_feed = Tool.OnehotEncode(label[:, 0], Loader.NUM_CATEGORIES)

            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            #### Test One Iteration
            loss_mb, Z_prob_mb, Z_mb = \
                self.sess.run([self.loss, self.Z_prob, self.Z],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[0:mb_size, ...]

            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)
                ## IoU
                avg_iou = Eval.EvalIoU(pred, seg[b_i], iou_oids)
                perdata_miou = (perdata_miou * data_cnt + avg_iou) / (data_cnt + 1)
                tmp = (pershape_miou[shape_label] * shape_cnt[shape_label] + avg_iou) / (shape_cnt[shape_label] + 1)
                pershape_miou[shape_label] = tmp
                ## Accuracy
                avg_acc = (avg_acc * data_cnt + np.mean(pred == seg[b_i])) / (data_cnt + 1)
                ## Loss
                avg_loss = (avg_loss * data_cnt + loss_mb) / (data_cnt + 1)

                data_cnt += 1
                shape_cnt[shape_label] += 1

            print('\rEvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Acc {:.3f}%  Avg IoU {:.3f}%'.format(
                data_cnt, avg_loss, 100 * avg_acc, 100 * np.mean(pershape_miou)), end='')


        return avg_loss, avg_acc, perdata_miou, pershape_miou

        ### Evaluation on validation set function

    def Test(self, Loader, Eval):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)

        while True:

            ## get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextSamp_TestSet()

            if not SuccessFlag:
                break

            #### Resample for fixed number of points
            pts_idx = np.arange(0, data.shape[1])
            add_resamp_idx = np.random.choice(pts_idx, 3000 - len(pts_idx), True)
            resamp_idx = np.concatenate([pts_idx, add_resamp_idx])

            data_feed = data[:, resamp_idx, :]
            seg_Onehot_feed = Tool.OnehotEncode(seg[:, resamp_idx], 50)
            label_Onehot_feed = Tool.OnehotEncode(label[:,0], Loader.NUM_CATEGORIES)
            # label_Onehot_feed = Tool.OnehotEncode(np.zeros([data_feed.shape[0]], dtype=int), Loader.NUM_CATEGORIES)
            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            #### Test One Iteration
            loss_mb, Z_prob_mb = \
                self.sess.run([self.loss, self.Z_prob],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[:, pts_idx, :]

            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)
                ## IoU
                # timer.tic()
                avg_iou = Eval.EvalIoU(pred, seg[b_i], iou_oids)
                # timer.toc()
                perdata_miou = (perdata_miou * data_cnt + avg_iou) / (data_cnt + 1)
                tmp = (pershape_miou[shape_label] * shape_cnt[shape_label] + avg_iou) / (shape_cnt[shape_label] + 1)
                pershape_miou[shape_label] = tmp
                ## Accuracy
                avg_acc = (avg_acc * data_cnt + np.mean(pred == seg[b_i])) / (data_cnt + 1)

                ## Loss
                avg_loss = (avg_loss * data_cnt + loss_mb) / (data_cnt + 1)
                # timer.tic()
                # iou, intersect, union = Tool.IoU_detail(pred[np.newaxis,:], seg[b_i][np.newaxis,:], Loader.NUM_PART_CATS)
                # np.mean(iou[0,iou_oids])
                # timer.toc()

                data_cnt += 1
                shape_cnt[shape_label] += 1

            print(
                '\rEvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Acc {:.3f}%  Avg PerData IoU {:.3f}%  Avg PerCat IoU {:.3f}%'.format(
                    data_cnt, avg_loss, 100 * avg_acc, 100 * np.mean(perdata_miou), 100 * np.mean(pershape_miou)),
                end='')

            # string = '\nEval PerShape IoU:'
            # for iou in pershape_miou:
            #     string += ' {:.2f}%'.format(100 * iou)
            # print(string)

            batch_cnt += 1

        return avg_loss, avg_acc, perdata_miou, pershape_miou

    ### Saving Checkpoint function
    def SaveCheckPoint(self, save_filepath, best_filename, eval_avg_correct_rate):

        save_filepath = os.path.abspath(save_filepath)

        self.saver.save(self.sess, save_filepath)

        filename = save_filepath.split('/')[-1]
        path = os.path.join(*save_filepath.split('/')[0:-1])
        path = '/' + path

        ## Save a copy of best performing model on validation set
        if self.bestValCorrect < np.mean(eval_avg_correct_rate):
            self.bestValCorrect = np.mean(eval_avg_correct_rate)

            src_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.data-00000-of-00001'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.index'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.index'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.index'.format(best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.meta'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.meta'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.meta'.format(best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

    ### Restore Checkpoint function
    def RestoreCheckPoint(self, filepath):

        self.saver.restore(self.sess, filepath)

    def WeakSupLoss(self):
        '''
        Define additional losses for weakly supervised segmentation
        Returns:

        '''

        ## Siamese Branch
        self.loss_siamese = tf.reduce_mean(
            tf.reduce_sum((self.Z_prob[0::2] - self.Z_prob[1::2]) ** 2, axis=-1))

        ## Inexact Supervision Branch
        L_gt = tf.cast(tf.reduce_max(self.Y_ph, axis=1), tf.float32)  # inexact labels B*50
        self.L = tf.reduce_max(self.Z, axis=1)  # B*50
        loss_ineaxct = tf.nn.sigmoid_cross_entropy_with_logits(labels=L_gt, logits=self.L)  # B*50
        self.loss_inexact = tf.reduce_mean(loss_ineaxct)

        ## Smooth Branch
        self.loss_smooth = 1e-1*SmoothConstraint.Loss_SpatialColorSmooth_add_SelfContain(self.Z_prob, self.X_ph)
