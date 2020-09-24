import numpy as np
import os
import h5py
import copy
import pathlib

class S3DIS_IO():

    ##### Data
    PartSeg = None
    numParts = None
    data = None
    label = None
    seg = None
    batch_sample_counter = 0
    finish_allbatch_flag = False
    CategoryName = None

    #### Methods
    def __init__(self,h5filepath='./',numParts = 13,batchsize = 24, NUM_POINT=4096):

        def getDataFiles(list_filename):
            return [line.rstrip() for line in open(list_filename)]

        self.data_base_path = h5filepath
        self.numParts = numParts
        self.ALL_FILES = getDataFiles(os.path.join(h5filepath,'all_files.txt'))
        self.room_filelist = [line.rstrip() for line in open(
            os.path.join(h5filepath,'room_filelist.txt'))]
        self.batchsize = batchsize
        self.NUM_POINT = NUM_POINT
        self.NUM_PART_CATS = 13

        self.NUM_CATEGORIES = self.NUM_PART_CATS

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename,'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        return (data, label)

    def loadDataFile(self, filename):
        return self.load_h5(filename)


    def LoadS3DIS_AllData(self):

        # Load ALL data
        data_batch_list = []
        label_batch_list = []
        for h5_filename in self.ALL_FILES:
            print('\rLoad file {}'.format(h5_filename),end='')
            data_batch, label_batch = self.loadDataFile(os.path.join(self.data_base_path,h5_filename.split('/')[1]))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        self.data_batches = np.concatenate(data_batch_list, 0)
        self.label_batches = np.concatenate(label_batch_list, 0)

    def CreateDataSplit(self,test_area):

        test_area = 'Area_' + str(test_area)
        train_idxs = []
        test_idxs = []
        all_idxs = []
        for i, room_name in enumerate(self.room_filelist):
            all_idxs.append(i)
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        self.train_data_idxs = np.array(train_idxs)
        # self.train_label = self.label_batches[train_idxs]
        self.test_data_idxs = np.array(test_idxs)
        # self.test_label = self.label_batches[test_idxs]
        self.all_data_idxs = np.array(all_idxs)

        self.train_samp_ptr = 0
        self.test_samp_ptr = 0
        self.all_samp_ptr = 0
        # print(data_batches.shape)
        # print(self.label_batches.shape)

    def ResetLoader_TrainSet(self):

        self.train_samp_ptr = 0
        self.shuffled_train_data_idxs = copy.deepcopy(self.train_data_idxs)

    def Shuffle_TrainSet(self):

        self.ResetLoader_TrainSet()
        np.random.shuffle(self.shuffled_train_data_idxs)


    def NextBatch_TrainSet(self):

        if self.train_samp_ptr + self.batchsize < self.shuffled_train_data_idxs.shape[0]:
            batch_idx = np.arange(self.train_samp_ptr, self.train_samp_ptr+self.batchsize)
            self.train_samp_ptr += self.batchsize
            mb_size = self.batchsize

        elif self.train_samp_ptr < self.shuffled_train_data_idxs.shape[0]:
            batch_idx = np.arange(self.train_samp_ptr, self.shuffled_train_data_idxs.shape[0])
            self.train_samp_ptr += self.batchsize
            mb_size = batch_idx.shape[0]
            # self.ResetLoader_TrainSet()
        else:
            # finished current epoch
            return False, None, None, None, None


        ## Collect data and segmentation labels
        data_idx = self.shuffled_train_data_idxs[batch_idx]
        data = copy.deepcopy(self.data_batches[data_idx])
        seg = copy.deepcopy(self.label_batches[data_idx])
        weak_seg_onehot = np.zeros([data.shape[0],self.numParts])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i,j] = 1
        ##
        # data ~ B*N*9  0:3 xyz coordinate with xy centered, 3:6 rgb values (divided by 255) and 6:9 xyz coordinate normalized to (0,0,0) to (1,1,1) cubic

        return True, data, seg, weak_seg_onehot, mb_size


    def NextBatch_TrainSet_v1(self):

        if self.train_samp_ptr + self.batchsize < self.shuffled_train_data_idxs.shape[0]:
            batch_idx = np.arange(self.train_samp_ptr, self.train_samp_ptr+self.batchsize)
            self.train_samp_ptr += self.batchsize
            mb_size = self.batchsize

        elif self.train_samp_ptr < self.shuffled_train_data_idxs.shape[0]:
            batch_idx = np.arange(self.train_samp_ptr, self.shuffled_train_data_idxs.shape[0])
            self.train_samp_ptr += self.batchsize
            mb_size = batch_idx.shape[0]
            # self.ResetLoader_TrainSet()
        else:
            # finished current epoch
            return False, None, None, None, None, None


        ## Collect data and segmentation labels
        data_idx = self.shuffled_train_data_idxs[batch_idx]
        data = copy.deepcopy(self.data_batches[data_idx])
        seg = copy.deepcopy(self.label_batches[data_idx])
        weak_seg_onehot = np.zeros([data.shape[0],self.numParts])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i,j] = 1


        return True, data, seg, weak_seg_onehot, mb_size, data_idx

    ##### Obtain next batch for all train & val sets
    def NextBatch_TrainValSet(self):

        if self.all_samp_ptr + self.batchsize < self.all_data_idxs.shape[0]:
            batch_idx = np.arange(self.all_samp_ptr, self.all_samp_ptr+self.batchsize)
            self.all_samp_ptr += self.batchsize
            mb_size = self.batchsize

        elif self.all_samp_ptr < self.all_data_idxs.shape[0]:
            batch_idx = np.arange(self.all_samp_ptr, self.all_data_idxs.shape[0])
            self.all_samp_ptr += self.batchsize
            mb_size = batch_idx.shape[0]
            # self.ResetLoader_TrainSet()
        else:
            # finished current epoch
            return False, None, None, None, None, None


        ## Collect data and segmentation labels
        data_idx = self.all_data_idxs[batch_idx]
        data = copy.deepcopy(self.data_batches[data_idx])
        seg = copy.deepcopy(self.label_batches[data_idx])
        weak_seg_onehot = np.zeros([data.shape[0],self.numParts])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i,j] = 1


        return True, data, seg, weak_seg_onehot, mb_size, data_idx


    ####### Functions for test set
    def NextBatch_TestSet(self,batchsize=None):

        if batchsize == None:
            batchsize = self.batchsize

        if self.test_samp_ptr + batchsize < self.test_data_idxs.shape[0]:
            batch_idx = np.arange(self.test_samp_ptr, self.test_samp_ptr+batchsize)
            self.test_samp_ptr += batchsize
            mb_size = batchsize
        elif self.test_samp_ptr < self.test_data_idxs.shape[0]:
            batch_idx = np.arange(self.test_samp_ptr, self.test_data_idxs.shape[0])
            self.test_samp_ptr += self.batchsize
            mb_size = batch_idx.shape[0]
        else:
            # finished current epoch
            return False, None, None, None, None


        ## Collect data and segmentation labels
        data = copy.deepcopy(self.data_batches[self.test_data_idxs[batch_idx]])
        seg = copy.deepcopy(self.label_batches[self.test_data_idxs[batch_idx]])
        weak_seg_onehot = np.zeros([data.shape[0],self.numParts])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i,j] = 1


        return True, data, seg, weak_seg_onehot, mb_size

    ####### Functions for test set
    def NextBatch_TestSet_v1(self, batchsize=None):

        if batchsize == None:
            batchsize = self.batchsize

        if self.test_samp_ptr + batchsize < self.test_data_idxs.shape[0]:
            batch_idx = np.arange(self.test_samp_ptr, self.test_samp_ptr + batchsize)
            self.test_samp_ptr += batchsize
            mb_size = batchsize
        elif self.test_samp_ptr < self.test_data_idxs.shape[0]:
            batch_idx = np.arange(self.test_samp_ptr, self.test_data_idxs.shape[0])
            self.test_samp_ptr += self.batchsize
            mb_size = batch_idx.shape[0]
        else:
            # finished current epoch
            return False, None, None, None, None, None

        ## Collect data and segmentation labels
        data_idx = self.test_data_idxs[batch_idx]
        data = copy.deepcopy(self.data_batches[data_idx])
        seg = copy.deepcopy(self.label_batches[data_idx])
        weak_seg_onehot = np.zeros([data.shape[0], self.numParts])
        for i in range(data.shape[0]):
            for j in np.unique(seg[i]):
                weak_seg_onehot[i, j] = 1

        return True, data, seg, weak_seg_onehot, mb_size, data_idx



    def ResetLoader_TestSet(self):

        self.test_samp_ptr = 0



class S3DIS_Test():

    def __init__(self, te_area, NUM_POINT=4096):
        self.te_area = te_area
        self.NUM_POINT = NUM_POINT

        ## Load All Room Path
        TE_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(),'../Dataset/S3DIS/')
        room_data_filelist = os.path.join(TE_DATA_PATH,'/meta/{}_data_label.txt'.format(self.te_area))
        self.ROOM_PATH_LIST = [os.path.join(TE_DATA_PATH, line.rstrip()) for line in open(room_data_filelist)]

        ## Reset Test Room Pointer
        self.ResetTestRoom()

        pass

    def ResetTestRoom(self):

        self.te_room_ptr = 0


    def LoadNextTestRoomData(self):

        if self.te_room_ptr >= len(self.ROOM_PATH_LIST):
            return None, None

        room_path = self.ROOM_PATH_LIST[self.te_room_ptr]
        current_data, current_label = self.room2blocks_wrapper_normalized(room_path, self.NUM_POINT)

        ## Increase test room pointer by 1
        self.te_room_ptr += 1

        return current_data, current_label

    def LoadNextTestRoomData_v1(self):

        if self.te_room_ptr >= len(self.ROOM_PATH_LIST):
            return None, None, None

        room_path = self.ROOM_PATH_LIST[self.te_room_ptr]
        current_data, current_label = self.room2blocks_wrapper_normalized(room_path, self.NUM_POINT)

        ## Increase test room pointer by 1
        self.te_room_ptr += 1

        return current_data, current_label, room_path

    def room2blocks_wrapper_normalized(self,data_label_filename, num_point, block_size=1.0, stride=1.0,
                                       random_sample=False, sample_num=None, sample_aug=1):
        if data_label_filename[-3:] == 'txt':
            data_label = np.loadtxt(data_label_filename)
        elif data_label_filename[-3:] == 'npy':
            data_label = np.load(data_label_filename)
        else:
            print('Unknown file type! exiting.')
            exit()
        return self.room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                           random_sample, sample_num, sample_aug)


    def room2blocks_plus_normalized(self, data_label, num_point, block_size, stride,
                                    random_sample, sample_num, sample_aug):
        """ room2block, with input filename and RGB preprocessing.
            for each block centralize XYZ, add normalized XYZ as 678 channels
        """
        data = data_label[:, 0:6]
        data[:, 3:6] /= 255.0
        label = data_label[:, -1].astype(np.uint8)
        max_room_x = max(data[:, 0])
        max_room_y = max(data[:, 1])
        max_room_z = max(data[:, 2])

        data_batch, label_batch = self.room2blocks(data, label, num_point, block_size, stride,
                                              random_sample, sample_num, sample_aug)
        new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
        for b in range(data_batch.shape[0]):
            new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
            new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
            new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx + block_size / 2)
            data_batch[b, :, 1] -= (miny + block_size / 2)
        new_data_batch[:, :, 0:6] = data_batch
        return new_data_batch, label_batch



    def room2blocks(self, data, label, num_point, block_size=1.0, stride=1.0,
                    random_sample=False, sample_num=None, sample_aug=1):
        """ Prepare block training data.
        Args:
            data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
                assumes the data is shifted (min point is origin) and aligned
                (aligned with XYZ axis)
            label: N size uint8 numpy array from 0-12
            num_point: int, how many points to sample in each block
            block_size: float, physical size of the block in meters
            stride: float, stride for block sweeping
            random_sample: bool, if True, we will randomly sample blocks in the room
            sample_num: int, if random sample, how many blocks to sample
                [default: room area]
            sample_aug: if random sample, how much aug
        Returns:
            block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
            block_labels: K x num_point x 1 np array of uint8 labels

        TODO: for this version, blocking is in fixed, non-overlapping pattern.
        """
        assert (stride <= block_size)

        limit = np.amax(data, 0)[0:3]

        # Get the corner location for our sampling blocks
        xbeg_list = []
        ybeg_list = []
        if not random_sample:
            num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
            num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(i * stride)
                    ybeg_list.append(j * stride)
        else:
            num_block_x = int(np.ceil(limit[0] / block_size))
            num_block_y = int(np.ceil(limit[1] / block_size))
            if sample_num is None:
                sample_num = num_block_x * num_block_y * sample_aug
            for _ in range(sample_num):
                xbeg = np.random.uniform(-block_size, limit[0])
                ybeg = np.random.uniform(-block_size, limit[1])
                xbeg_list.append(xbeg)
                ybeg_list.append(ybeg)

        # Collect blocks
        block_data_list = []
        block_label_list = []
        idx = 0
        for idx in range(len(xbeg_list)):
            xbeg = xbeg_list[idx]
            ybeg = ybeg_list[idx]
            xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
            ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
            cond = xcond & ycond
            if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
                continue

            block_data = data[cond, :]
            block_label = label[cond]

            # randomly subsample data
            block_data_sampled, block_label_sampled = \
                self.sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))

        return np.concatenate(block_data_list, 0), \
               np.concatenate(block_label_list, 0)

    def sample_data_label(self,data, label, num_sample):
        new_data, sample_indices = self.sample_data(data, num_sample)
        new_label = label[sample_indices]
        return new_data, new_label

    def sample_data(self,data, num_sample):
        """ data is in N x ...
            we want to keep num_samplexC of them.
            if N > num_sample, we will randomly keep num_sample of them.
            if N < num_sample, we will randomly duplicate samples.
        """
        N = data.shape[0]
        if (N == num_sample):
            return data, range(N)
        elif (N > num_sample):
            sample = np.random.choice(N, num_sample)
            return data[sample, ...], sample
        else:
            sample = np.random.choice(N, num_sample - N)
            dup_data = data[sample, ...]
            return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)