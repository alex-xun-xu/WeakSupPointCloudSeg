######### Utility functions for S3DIS dataset
import numpy as np
import tensorflow as tf
import os
import sys
import scipy.io as scio

sys.path.append(os.path.expanduser('../Util'))

import Tool
import DGCNN_S3DIS as network
import SmoothConstraint
import ProbLabelPropagation as PLP


class S3DIS_Trainer():

    def __init__(self, test_area):

        self.bestValCorrect = 0.    # initial best validation performance

        pass

    def SetLearningRate(self, LearningRate, BatchSize):

        self.BASE_LEARNING_RATE = LearningRate
        self.BATCH_SIZE = BatchSize
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.DECAY_STEP = 300000
        self.DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP * 2)
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


    def defineNetwork(self, batch_size, num_points, style='Full', rampup=101):
        '''
        define DGCNN network for incomplete labels as supervision
        Args:
            batch_size: batchsize for training network
            num_points: number of points for each point cloud sample
            style: model style, use full model or plain model
            rampup: rampup epoch for training
        :return:
        '''

        ##### Parameters
        self.rampup = rampup
        self.style = style

        ##### Define Network Inputs
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_points, 9], name='InputPts')  # B*N*3
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_points, 13], name='PartGT')  # B*N*13
        self.Mask_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_points], name='Mask')  # B*N
        self.Is_Training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='IsTraining')

        ## Set up batch norm decay and learning rate decay
        self.batch = tf.Variable(0, trainable=False)
        bn_decay = self.get_bn_decay()
        learning_rate = self.get_learning_rate()

        ##### Define DGCNN network
        self.Z = network.get_model(self.X_ph, self.Is_Training_ph, weight_decay=0., bn_decay=bn_decay)

        self.Z_exp = tf.exp(self.Z)
        self.Z_prob = tf.nn.softmax(self.Z, axis=-1)

        ## Segmentation Branch
        loss_seg = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_ph, logits=self.Z)  # B*N
        self.loss_seg = tf.reduce_sum(self.Mask_ph * loss_seg) / tf.reduce_sum(self.Mask_ph)

        ## Final Loss
        self.epoch = 0
        if style == 'Plain':
            # plain style training - only the labeled points are used for supervision
            self.loss = self.loss_seg
        elif style == 'Full':
            # full style training - all weakly supervised losses are used for training
            self.WeakSupLoss()
            self.loss = self.loss_seg + \
                        tf.cast(tf.greater_equal(self.epoch, self.rampup), dtype=tf.float32) * (
                                    self.loss_siamese + self.loss_inexact + self.loss_smooth)
        else:
            sys.exit('Loss {} is not defined!'.format(self.loss))

        ## Final Loss
        # self.loss = self.loss_seg + self.loss_siamese + self.loss_inexact + self.loss_smooth

        ##### Define Optimizer
        self.solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                   global_step=self.batch)  # initialize solver
        self.saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        return True

    def WeakSupLoss(self):
        '''
        Define additional losses for weakly supervised segmentation
        Returns:

        '''

        ## Siamese Branch
        self.loss_siamese = 1e1 * tf.reduce_mean(tf.reduce_sum((self.Z_prob[0::2] - self.Z_prob[1::2]) ** 2, axis=-1))

        ## MIL Branch
        L_gt = tf.cast(tf.reduce_max(self.Y_ph, axis=1), tf.float32)  # inexact labels B*13
        self.L = tf.reduce_max(self.Z, axis=1)
        loss_ineaxct = tf.nn.sigmoid_cross_entropy_with_logits(labels=L_gt, logits=self.L)  # B*K
        self.loss_inexact = tf.reduce_mean(loss_ineaxct)

        ## Smooth Branch
        self.loss_smooth = SmoothConstraint.Loss_SpatialColorSmooth_add_SelfContain(self.Z_prob, self.X_ph[:, :, 0:6])

    def defLabelPropSolver(self,alpha=1e0,beta=1e0,K=10):
        ##### Define Label Propagation Solver
        self.LPSolver = PLP.LabelPropagation_Baseline_TF(alpha=1e0, beta=1e0, K=10)
        self.TFComp = {}
        self.TFComp['Lmat'] = Tool.TF_Computation.LaplacianMatSym_XYZRGB_DirectComp()


    ##### Define Training and Evaluation functions
    def TrainOneEpoch(self, Loader, pts_idx_list=None, batch_size=12):
        '''
        Function to train one epoch
        :param Loader: Object to load training data
        :param samp_idx_list:  A list indicating the labelled points  B*N
        :return:
        '''
        batch_cnt = 1
        data_cnt = 0
        # shape_cnt = np.zeros(shape=[len(Loader.objcats)])
        avg_loss = 0.
        avg_acc = 0.

        while True:

            #### get next batch
            SuccessFlag, data, seg, weak_seg_onehot, mb_size, data_idx = Loader.NextBatch_TrainSet_v1()

            if not SuccessFlag:
                break

            if mb_size < batch_size:
                break

            #### Prepare Incomplete Labelled Training Data
            if pts_idx_list is None:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)  # B*N
            else:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)  # B*N
                for b_i in range(mb_size):
                    batch_samp_idx = data_idx[b_i]
                    Mask_bin[b_i, pts_idx_list[batch_samp_idx]] = 1
            Mask_bin_feed = Mask_bin

            #### Create Siamese Input
            data_feed = data

            #### Prepare Labels
            seg_onehot_feed = Tool.OnehotEncode(seg, 13)

            #### Train One Iteration
            _, loss_mb, Z_prob_mb = \
                self.sess.run(
                    [self.solver, self.loss, self.Z_prob],
                    feed_dict={self.X_ph: data_feed,
                               self.Y_ph: seg_onehot_feed,
                               self.Is_Training_ph: True,
                               self.Mask_ph: Mask_bin_feed})

            ## Calculate loss and correct rate
            avg_loss = (avg_loss * data_cnt + loss_mb * mb_size) / (data_cnt + mb_size)
            pred = []
            for b_i in range(mb_size):
                pred.append(np.argmax(Z_prob_mb[b_i], axis=-1))
            pred = np.stack(pred)
            avg_acc = (avg_acc * data_cnt + np.mean(pred == seg) * mb_size) / (data_cnt + mb_size)

            data_cnt += mb_size

            print(
                '\rBatch {:d} TrainedSamp {:d}  mbLoss {:.4f} '
                'Avg Acc {:.2f}%'.format(
                    batch_cnt, data_cnt, loss_mb, 100 * avg_acc),
                end='')

            batch_cnt += 1

        Loader.ResetLoader_TrainSet()

        # increase epoch counter by 1
        self.epoch += 1

        # return avg_loss, perdata_miou, pershape_miou
        return avg_loss, avg_acc

    def TrainOneEpoch_Full(self, Loader, pts_idx_list=None, batch_size=12):
        '''
        Function to train one epoch
        :param Loader: Object to load training data
        :param samp_idx_list:  A list indicating the labelled points  B*N
        :return:
        '''
        batch_cnt = 1
        data_cnt = 0
        # shape_cnt = np.zeros(shape=[len(Loader.objcats)])
        avg_loss = 0.
        avg_acc = 0.

        while True:

            #### get next batch
            SuccessFlag, data, seg, weak_seg_onehot, mb_size, data_idx = Loader.NextBatch_TrainSet_v1()

            if not SuccessFlag or batch_cnt > np.inf:
                break

            if mb_size < batch_size:
                break

            #### Prepare Incomplete Labelled Training Data
            if pts_idx_list is None:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)  # B*N
            else:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)  # B*N
                for b_i in range(mb_size):
                    batch_samp_idx = data_idx[b_i]
                    Mask_bin[b_i, pts_idx_list[batch_samp_idx]] = 1

            Mask_bin_feed = []
            for mask_i in Mask_bin:
                Mask_bin_feed.append(mask_i)
                Mask_bin_feed.append(mask_i)
            Mask_bin_feed = np.stack(Mask_bin_feed)

            #### Create Siamese Input
            if self.epoch >= self.rampup:
                data_feed = []
                for data_i in data:
                    data_feed.append(data_i)
                    aug_choice = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], 1)
                    if aug_choice == 1:
                        data_i[:, 0], data_i[:, 1] = data_i[:, 1].copy(), data_i[:, 0].copy()
                        data_i[:, 6], data_i[:, 7] = data_i[:, 7].copy(), data_i[:, 6].copy()
                    elif aug_choice == 2:
                        data_i[:, 0] = -data_i[:, 0]
                        data_i[:, 6] = -data_i[:, 6] + 1
                    elif aug_choice == 3:
                        data_i[:, 1] = -data_i[:, 1]
                        data_i[:, 7] = -data_i[:, 7] + 1
                    elif aug_choice == 4:
                        data_i[:, 0] = -data_i[:, 0]
                        data_i[:, 6] = -data_i[:, 6] + 1
                        data_i[:, 1] = -data_i[:, 1]
                        data_i[:, 7] = -data_i[:, 7] + 1
                    elif aug_choice == 5:
                        data_i[:, 0], data_i[:, 1] = data_i[:, 1].copy(), data_i[:, 0].copy()
                        data_i[:, 6], data_i[:, 7] = data_i[:, 7].copy(), data_i[:, 6].copy()
                        data_i[:, 0] = -data_i[:, 0]
                        data_i[:, 6] = -data_i[:, 6] + 1
                    elif aug_choice == 6:
                        data_i[:, 0], data_i[:, 1] = data_i[:, 1].copy(), data_i[:, 0].copy()
                        data_i[:, 6], data_i[:, 7] = data_i[:, 7].copy(), data_i[:, 6].copy()
                        data_i[:, 1] = -data_i[:, 1]
                        data_i[:, 7] = -data_i[:, 7] + 1
                    elif aug_choice == 7:
                        data_i[:, 0], data_i[:, 1] = data_i[:, 1].copy(), data_i[:, 0].copy()
                        data_i[:, 6], data_i[:, 7] = data_i[:, 7].copy(), data_i[:, 6].copy()
                        data_i[:, 0] = -data_i[:, 0]
                        data_i[:, 6] = -data_i[:, 6] + 1
                        data_i[:, 1] = -data_i[:, 1]
                        data_i[:, 7] = -data_i[:, 7] + 1

                    data_feed.append(data_i)

                data_feed = np.stack(data_feed, axis=0)
            else:
                data_feed = []
                for data_i in data:
                    data_feed.append(data_i)
                    data_feed.append(data_i)
                data_feed = np.stack(data_feed)

            ## Prepare Labels
            seg_onehot = Tool.OnehotEncode(seg, 13)
            seg_feed_onehot = []
            for seg_i in seg_onehot:
                seg_feed_onehot.append(seg_i)
                seg_feed_onehot.append(seg_i)
            seg_feed_onehot = np.stack(seg_feed_onehot)

            #### Train One Iteration
            _, loss_mb, loss_siamese_mb, loss_mil_mb, loss_smooth_mb, Z_prob_mb = \
                self.sess.run(
                    [self.solver, self.loss, self.loss_siamese, self.loss_inexact, self.loss_smooth, self.Z_prob],
                    feed_dict={self.X_ph: data_feed,
                               self.Y_ph: seg_feed_onehot,
                               self.Is_Training_ph: True,
                               self.Mask_ph: Mask_bin_feed})

            ## Calculate loss and correct rate
            avg_loss = (avg_loss * data_cnt + loss_mb * mb_size) / (data_cnt + mb_size)
            pred = []
            for b_i in range(mb_size):
                pred.append(np.argmax(Z_prob_mb[2 * b_i], axis=-1))
            pred = np.stack(pred)
            avg_acc = (avg_acc * data_cnt + np.mean(pred == seg) * mb_size) / (data_cnt + mb_size)

            data_cnt += mb_size

            print(
                '\rBatch {:d} TrainedSamp {:d}  mbLoss {:.4f} SiamLoss {:.3f} MILLoss {:.3f} SmoothLoss {:.3f} '
                'Avg Acc {:.2f}%'.format(
                    batch_cnt, data_cnt, loss_mb, loss_siamese_mb, loss_mil_mb, loss_smooth_mb, 100 * avg_acc),
                end='')

            batch_cnt += 1

        Loader.ResetLoader_TrainSet()

        # increase epoch counter by 1
        self.epoch += 1

        # return avg_loss, perdata_miou, pershape_miou
        return avg_loss, avg_acc

    def EvalOneEpoch(self, Loader):
        batch_cnt = 1
        samp_cnt = 0
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_iou = 0.
        while True:

            ## get next batch
            SuccessFlag, data, seg, weak_seg_onehot, mb_size = Loader.NextBatch_TestSet()

            if not SuccessFlag:
                break

            ## Normalize Validation Points
            # data = PointNet_CAM.pc_max_normalize(data)

            ## Dummy Variables
            Mask_bin = np.ones(shape=[mb_size, data.shape[1]], dtype=np.float32)

            loss_mb, Z_prob_mb = \
                self.sess.run([self.loss, self.Z_prob],
                              feed_dict={self.X_ph: data,
                                         self.Y_ph: seg,
                                         self.Is_Training_ph: True,
                                         self.Mask_ph: Mask_bin})

            ## Calculate loss and correct rate
            pred_mb = np.argmax(Z_prob_mb, axis=-1)
            correct = np.mean(pred_mb == seg)
            m_iou = np.mean(Tool.IoU(pred_mb, seg, Loader.numParts))

            # avg_loss = (batch_cnt - 1) / batch_cnt * avg_loss + (loss_mb / mb_size) / batch_cnt
            # avg_correct_rate = (batch_cnt - 1) / batch_cnt * avg_correct_rate + correct / batch_cnt

            avg_loss = (avg_loss * samp_cnt + loss_mb) / (samp_cnt + mb_size)
            avg_correct_rate = (avg_correct_rate * samp_cnt + correct * mb_size) / (samp_cnt + mb_size)
            avg_iou = (avg_iou * samp_cnt + m_iou * mb_size) / (samp_cnt + mb_size)

            samp_cnt += mb_size

            print('\rBatch {:d} EvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Correct Rate {:.3f}%  Avg IoU {:.3f}%'.format(
                batch_cnt, samp_cnt, avg_loss, 100 * avg_correct_rate, 100 * avg_iou), end='')

            batch_cnt += 1

        Loader.ResetLoader_TestSet()

        return avg_loss, avg_correct_rate, avg_iou

    def EvalOneEpoch_Full(self, Loader):
        '''
        Evaluate the full model for one epoch
        Args:
            Loader: Data loader object

        Returns:

        '''
        batch_cnt = 1
        samp_cnt = 0
        true_positive_classes = np.zeros(shape=[13])
        positive_classes = np.zeros(shape=[13])
        gt_classes = np.zeros(shape=[13])
        total_correct = 0.
        total_seen = 0.
        avg_loss = 0.
        avg_correct_rate = 0.
        iou = 0.

        while True:

            ## get next batch
            SuccessFlag, data, seg_mb, weak_seg_onehot, mb_size = Loader.NextBatch_TestSet()

            if not SuccessFlag:
                break

            if mb_size < Loader.batchsize:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [Loader.batchsize - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg_mb, np.tile(seg_mb[np.newaxis, 0], [Loader.batchsize - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 13)
            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg_mb, 13)

            Mask_bin_feed = np.ones(shape=[Loader.batchsize, data.shape[1]], dtype=np.float32)

            ## Replicate for Siamese Network Input
            data_feed_rep = []
            seg_Onehot_feed_rep = []
            Mask_bin_feed_rep = []

            for b_i in range(self.BATCH_SIZE):
                data_feed_rep.append(data_feed[b_i])
                data_feed_rep.append(data_feed[b_i])
                seg_Onehot_feed_rep.append(seg_Onehot_feed[b_i])
                seg_Onehot_feed_rep.append(seg_Onehot_feed[b_i])
                Mask_bin_feed_rep.append(Mask_bin_feed[b_i])
                Mask_bin_feed_rep.append(Mask_bin_feed[b_i])

            data_feed_rep = np.stack(data_feed_rep, axis=0)
            seg_Onehot_feed_rep = np.stack(seg_Onehot_feed_rep, axis=0)
            Mask_bin_feed_rep = np.stack(Mask_bin_feed_rep, axis=0)

            loss_mb, Z_prob_mb = \
                self.sess.run([self.loss, self.Z_prob],
                              feed_dict={self.X_ph: data_feed_rep,
                                         self.Y_ph: seg_Onehot_feed_rep,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed_rep})

            Z_prob_mb = Z_prob_mb[0:2 * mb_size:2, ...]

            ## Calculate loss and correct rate
            pred_mb = np.argmax(Z_prob_mb, axis=-1)
            correct = np.sum(pred_mb == seg_mb)
            acc = np.mean(pred_mb == seg_mb)
            # m_iou = np.mean(Tool.IoU(pred_mb, seg, Loader.numParts))
            total_correct += correct
            total_seen += (mb_size * Loader.NUM_POINT)
            for pred, label in zip(pred_mb,seg_mb):
                for pt_i in range(Loader.NUM_POINT):
                    positive_classes[pred[pt_i]] += 1
                    true_positive_classes[label[pt_i]] += float(pred[pt_i] == label[pt_i])
                    gt_classes[label[pt_i]] += 1

            ## Calculate IoU
            iou = true_positive_classes / (
                    gt_classes + positive_classes - true_positive_classes + 1e-5)

            avg_loss = (avg_loss * samp_cnt + loss_mb) / (samp_cnt + mb_size)
            avg_correct_rate = (avg_correct_rate * samp_cnt + acc * mb_size) / (samp_cnt + mb_size)
            # avg_iou = (avg_iou * samp_cnt + m_iou * mb_size) / (samp_cnt + mb_size)

            samp_cnt += mb_size

            print('\rBatch {:d} EvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Correct Rate {:.3f}%  mIoU {:.3f}%'.format(
                batch_cnt, samp_cnt, avg_loss, 100 * avg_correct_rate, 100 * np.mean(iou)), end='')

            batch_cnt += 1

        Loader.ResetLoader_TestSet()


        return avg_loss, avg_correct_rate, np.mean(iou)


    def Test(self, Loader, PRED_PATH):
        '''
        Inference on test set
        Args:
            Loader: Data loader object
            PRED_PATH: the path to save inference results

        Returns:

        '''
        true_positive_classes = np.zeros(shape=[13])
        positive_classes = np.zeros(shape=[13])
        gt_classes = np.zeros(shape=[13])
        total_correct = 0.
        total_seen = 0.
        avg_loss = 0.
        samp_cnt = 0
        room_cnt = 0

        while True:

            ## Evaluate Each Room
            data, label, room_path = Loader.LoadNextTestRoomData_v1()

            if data is None:
                break

            ## Do inference block by block
            allPred = []
            allGT = []

            for blk_i, data_i, label_i in zip(range(data.shape[0]), data, label):

                ## Inference One Block
                feed_dict = {self.X_ph: data_i[np.newaxis, ...],
                             self.Y_ph: Tool.OnehotEncode(label_i[np.newaxis, ...], 13),
                             self.Mask_ph: np.ones([1, data_i.shape[0]]),
                             self.Is_Training_ph: False}

                loss_mb, Z_prob_mb = \
                    self.sess.run([self.loss, self.Z_prob],
                                  feed_dict=feed_dict)

                ## Apply Label Propagation
                Lmat = self.TFComp['Lmat'].Eval(self.sess, data_i[np.newaxis, :, 0:3], data_i[np.newaxis, :, 3:6])
                _, Z_prob_LP, w = self.LPSolver.SolveLabelProp(self.sess, Lmat[0], Z_prob_mb[0])

                ## Evaluate Performance
                avg_loss = (avg_loss * blk_i + loss_mb * 1) / (blk_i + 1)
                pred = np.argmax(Z_prob_LP, axis=-1)
                correct = np.sum(pred == label_i)
                total_correct += correct
                total_seen += (1 * Loader.NUM_POINT)
                for pt_i in range(Loader.NUM_POINT):
                    positive_classes[pred[pt_i]] += 1
                    true_positive_classes[label_i[pt_i]] += float(pred[pt_i] == label_i[pt_i])
                    gt_classes[label_i[pt_i]] += 1

                samp_cnt += 1

                ## Accuracy
                acc = total_correct / total_seen

                ## Calculate IoU
                iou = true_positive_classes / (
                        gt_classes + positive_classes - true_positive_classes + 1e-5)

                ## Append predictions
                allPred.append(pred)
                allGT.append(label_i)

                print('\rroom {:d}  acc {:.2f}%  iou: {:.2f}%'.format(room_cnt, 100 * acc, 100 * np.mean(iou)), end='')

            #### Save Predictions for Current Room
            allPred = np.concatenate(allPred, axis=0)
            allGT = np.concatenate(allGT, axis=0)

            room_name = room_path.split('/')[-1].split('.')[0]

            room_pred_filepath = os.path.join(PRED_PATH, '{}_pred_gt.mat'.format(room_name))

            scio.savemat(room_pred_filepath, {'data': data, 'pred': allPred, 'gt': allGT})

            room_cnt += 1

        return true_positive_classes, positive_classes, gt_classes


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

    ########## Restore Checkpoint function
    def RestoreCheckPoint(self, filepath):

        self.saver.restore(self.sess, filepath)