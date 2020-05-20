import numpy as np
import os
import h5py
import copy
import sys
import json
import scipy.io as scio

class ShapeNetIO:

    def __init__(self, BASE_DIR='/vision01/pointnet/part_seg/', batchsize=24):

        self.BASE_DIR = BASE_DIR
        self.h5_base_path = os.path.join(self.BASE_DIR, 'hdf5_data')
        self.ply_data_dir = os.path.join(self.BASE_DIR, 'PartAnnotation')

        self.batchsize = batchsize
        self.color_map_file = os.path.join(self.h5_base_path, 'part_color_mapping.json')
        self.color_map = json.load(open(self.color_map_file, 'r'))

        self.all_obj_cats_file = os.path.join(self.h5_base_path, 'all_object_categories.txt')
        fin = open(self.all_obj_cats_file, 'r')
        lines = [line.rstrip() for line in fin.readlines()]
        self.all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
        fin.close()

        all_cats = json.load(open(os.path.join(self.h5_base_path, 'overallid_to_catid_partid.json'), 'r'))
        self.NUM_CATEGORIES = 16
        self.NUM_PART_CATS = len(all_cats)

        ## Load Part Categories for each shape
        oid2cpid = json.load(open(os.path.join(self.h5_base_path, 'overallid_to_catid_partid.json'), 'r'))

        self.cpid2oid = json.load(open(os.path.join(self.h5_base_path, 'catid_partid_to_overallid.json'), 'r'))

        self.object2setofoid = {}
        for idx in range(len(oid2cpid)):
            objid, pid = oid2cpid[idx]
            if not objid in self.object2setofoid.keys():
                self.object2setofoid[objid] = []
            self.object2setofoid[objid].append(idx)

        ## Shape Category ID
        all_obj_cat_file = os.path.join(self.h5_base_path, 'all_object_categories.txt')
        fin = open(all_obj_cat_file, 'r')
        lines = [line.rstrip() for line in fin.readlines()]
        self.objcats = [line.split()[1] for line in lines]
        self.objnames = [line.split()[0] for line in lines]
        self.on2oid = {self.objcats[i]: i for i in range(len(self.objcats))}
        fin.close()


    def LoadTrainValFiles(self):

        def getDataFiles(list_filename):
            return [line.rstrip() for line in open(list_filename)]

        self.TRAINING_FILE_LIST = os.path.join(self.h5_base_path, 'train_hdf5_file_list.txt')
        self.VAL_FILE_LIST = os.path.join(self.h5_base_path, 'val_hdf5_file_list.txt')

        ## train/val file list
        self.train_file_list = getDataFiles(self.TRAINING_FILE_LIST)
        self.num_train_file = len(self.train_file_list)
        self.val_file_list = getDataFiles(self.VAL_FILE_LIST)
        self.num_test_file = len(self.val_file_list)

        ## train/val file index
        self.train_file_idx = np.arange(0,len(self.train_file_list))
        self.val_file_idx = np.arange(0,len(self.val_file_list))

        ## Load Train Data
        train_data = []
        train_labels = []
        train_seg = []
        num_train = 0
        train_data_idx = []
        for cur_train_filename in self.train_file_list:
            cur_train_data, cur_train_labels, cur_train_seg, cur_num_train, cur_train_data_idx = self.loadDataFile_with_seg(
                os.path.join(self.h5_base_path,cur_train_filename))

            train_data.append(cur_train_data)
            train_labels.append(cur_train_labels)
            train_seg.append(cur_train_seg)
            train_data_idx.append(cur_train_data_idx+num_train)
            num_train += cur_num_train

        self.train_data = np.concatenate(train_data)
        self.train_labels = np.concatenate(train_labels)
        self.train_seg = np.concatenate(train_seg)
        self.train_data_idx = np.concatenate(train_data_idx)
        self.num_train = num_train

        ## Load Val Data
        val_data = []
        val_labels = []
        val_seg = []
        num_val = 0
        val_data_idx = []
        for cur_val_filename in self.val_file_list:
            cur_val_data, cur_val_labels, cur_val_seg, cur_num_val, cur_val_data_idx = self.loadDataFile_with_seg(
                os.path.join(self.h5_base_path, cur_val_filename))

            val_data.append(cur_val_data)
            val_labels.append(cur_val_labels)
            val_seg.append(cur_val_seg)
            val_data_idx.append(cur_val_data_idx + num_val)
            num_val += cur_num_val

        self.val_data = np.concatenate(val_data)
        self.val_labels = np.concatenate(val_labels)
        self.val_seg = np.concatenate(val_seg)
        self.val_data_idx = np.concatenate(val_data_idx)
        self.num_val = num_val

        ## Intialize train/val loader
        self.ResetLoader_TrainSet()
        self.ResetLoader_ValSet()

        # self.ValSet_Loaded_Flag = False


    def LoadTestFiles(self):

        def getDataFiles(list_filename):
            return [line.rstrip() for line in open(list_filename)]

        self.TEST_FILE_LIST = os.path.join(self.BASE_DIR, 'testing_ply_file_list.txt')

        ## test file list
        ffiles = open(self.TEST_FILE_LIST, 'r')
        lines = [line.rstrip() for line in ffiles.readlines()]
        self.test_pts_files = [line.split()[0] for line in lines]
        self.test_seg_files = [line.split()[1] for line in lines]
        self.test_labels = [line.split()[2] for line in lines]
        ffiles.close()

        ## test file index
        self.test_samp_num = len(self.test_pts_files)
        self.test_file_idx = np.arange(0,self.test_samp_num)

        ## Intialize test loader
        self.ResetLoader_TestSet()


    def NextBatch_TrainSet(self,shuffle_flag=False):
        '''

        :param shuffle_flag:
        :return:
        SuccessFlag
        data    the point cloud data dim~B*N*D
        label   the shape label dim~B
        seg     the segmentation label dim~B*N
        weak_seg_onehot     the weak segmentation label (onehot) dim~B*K
        mb_size     the minibatch size dim~1
        file_idx    the index of the train file dim~B (default all zeros to be compatible with older versions)
        data_idx    the index of the train sample dim~B
        '''

        if self.train_end_dataset:
            ## Reached the end of training dataset. Return false to break current training epoch
            self.ResetLoader_TrainSet()
            return False, None, None, None, None, None, None, None

        #### Obtain the next batch data
        if self.train_samp_ptr + self.batchsize < self.num_train:
            ## Not reached the end of current train data file
            batch_idx = np.arange(self.train_samp_ptr, self.train_samp_ptr+self.batchsize)
            self.train_samp_ptr += self.batchsize
            mb_size = self.batchsize
        elif self.train_samp_ptr < self.num_train:
            ## Reached the end of current train data file
            batch_idx = np.arange(self.train_samp_ptr, self.num_train)
            self.train_samp_ptr = self.num_train
            mb_size = batch_idx.shape[0]
            self.train_end_dataset = True
        else:
            ## Reached the end of current train data file
            self.ResetLoader_TrainSet()
            return False, None, None, None, None, None, None, None

        #### Collect data and segmentation labels
        data_idx = copy.deepcopy(self.train_data_idx[batch_idx])
        file_idx = np.zeros_like(data_idx)
        data = copy.deepcopy(self.train_data[data_idx])
        label = copy.deepcopy(self.train_labels[data_idx])
        seg = copy.deepcopy(self.train_seg[data_idx])
        weak_seg_onehot = np.zeros([mb_size, self.NUM_PART_CATS])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i,j] = 1

        return True, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx


    def NextBatch_ValSet(self):

        if self.val_end_dataset:
            ## Reached the end of valing dataset. Return false to break current valing epoch
            self.ResetLoader_ValSet()
            return False, None, None, None, None, None, None, None

        #### Obtain the next batch data
        if self.val_samp_ptr + self.batchsize < self.num_val:
            ## Not reached the end of current val data file
            batch_idx = np.arange(self.val_samp_ptr, self.val_samp_ptr + self.batchsize)
            self.val_samp_ptr += self.batchsize
            mb_size = self.batchsize
        elif self.val_samp_ptr < self.num_val:
            ## Reached the end of current val data file
            batch_idx = np.arange(self.val_samp_ptr, self.num_val)
            self.val_samp_ptr = self.num_val
            mb_size = batch_idx.shape[0]
            self.val_end_dataset = True
        else:
            ## Reached the end of current val data file
            self.ResetLoader_ValSet()
            return False, None, None, None, None, None, None, None


        #### Collect data and segmentation labels
        data_idx = copy.deepcopy(self.val_data_idx[batch_idx])
        file_idx = np.zeros_like(data_idx)
        data = copy.deepcopy(self.val_data[data_idx])
        label = copy.deepcopy(self.val_labels[data_idx])
        seg = copy.deepcopy(self.val_seg[data_idx])
        weak_seg_onehot = np.zeros([mb_size, self.NUM_PART_CATS])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i, j] = 1

        return True, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx


    def NextSamp_TestSet(self):

        ## Initial return variables
        SuccessFlag = False
        data = None
        label = None
        seg = None
        weak_seg_onehot = None
        mb_size = None
        file_idx = None
        data_idx = None

        ## Check reaching the end of test samples
        if self.te_samp_ptr >= self.test_samp_num:
            self.ResetLoader_TestSet()
            return SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx

        ## Load current testing sample
        cur_gt_label = self.on2oid[self.test_labels[self.te_samp_ptr]]
        cur_label_one_hot = np.zeros((1, self.NUM_CATEGORIES), dtype=np.float32)
        cur_label_one_hot[0, cur_gt_label] = 1

        pts_file_to_load = os.path.join(self.ply_data_dir, self.test_pts_files[self.te_samp_ptr])
        seg_file_to_load = os.path.join(self.ply_data_dir, self.test_seg_files[self.te_samp_ptr])

        pts, seg = self.load_pts_seg_files(pts_file_to_load, seg_file_to_load, self.objcats[cur_gt_label])
        ori_point_num = len(seg)

        pts = self.pc_normalize(pts)

        ## Prepare return variables
        SuccessFlag = True
        data = pts[np.newaxis,...]
        label = np.array([[int(cur_gt_label)]])
        seg = seg[np.newaxis,...]
        mb_size = 1
        weak_seg_onehot = np.zeros([mb_size, self.NUM_PART_CATS])
        for i in range(mb_size):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i, j] = 1
        file_idx = 0
        data_idx = self.te_samp_ptr

        ## Increase test sample pointer by 1
        self.te_samp_ptr += 1

        return SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx


    def Shuffle_TrainSet(self):

        np.random.shuffle(self.train_data_idx)


    def ResetLoader_TrainSet(self):

        ## train/val file pointer
        self.train_samp_ptr = 0

        ## end of current file
        self.train_end_dataset = False  # flag indicating reaching the end of current dataset


    def ResetLoader_ValSet(self):

        ## train/val file pointer
        self.val_samp_ptr = 0

        ## end of current file
        self.val_end_dataset = False  # flag indicating reaching the end of current dataset


    def ResetLoader_TestSet(self):

        ## test file pointer
        self.te_samp_ptr = 0


    def loadDataFile_with_seg(self,filename):
        return self.load_h5_data_label_seg(filename)


    def load_h5_data_label_seg(self,h5_filename):
        f = h5py.File(h5_filename,'r')
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        num_data = data.shape[0]
        data_idx = np.arange(0,num_data)

        return (data, label, seg, num_data, data_idx)


    def load_pts_seg_files(self,pts_file, seg_file, catid):
        with open(pts_file, 'r') as f:
            pts_str = [item.rstrip() for item in f.readlines()]
            pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
        with open(seg_file, 'r') as f:
            part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
            seg = np.array([self.cpid2oid[catid+'_'+str(x)] for x in part_ids])
        return pts, seg


    def pc_normalize(self,pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc
