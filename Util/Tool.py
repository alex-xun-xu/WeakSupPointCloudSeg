import numpy as np
import tensorflow as tf

def OnehotEncode(Y,K):
    "Onehot key encoding of input label Y"
    #
    #   Y ~ B*N
    #   K ~ 1   The largest category index
    #

    if len(Y.shape)>1:
        Y_onehot = np.zeros(shape=[Y.shape[0],Y.shape[1],K])

        for b in range(Y.shape[0]):
            for r in range(Y.shape[1]):

                Y_onehot[b,r,Y[b,r]] = 1

    elif len(Y.shape)==1:
        Y_onehot = np.zeros(shape=[Y.shape[0],K])
        for r in range(Y.shape[0]):
            Y_onehot[r, Y[r]] = 1

    else:
        Y_onehot = np.zeros(shape=[K])
        Y_onehot[Y] = 1

    return Y_onehot

def pdist2(X, Y):
    '''
    Compute Pairwise distance between X and Y
    :param X: B*N*D
    :param Y: B*M*D
    :return: D: B*N*M
    '''

    X2 = tf.expand_dims(tf.reduce_sum(X ** 2, axis=-1), axis=-1)  # B*N*1
    Y2 = tf.expand_dims(tf.reduce_sum(Y ** 2, axis=-1), axis=1)  # B*1*M
    XY = tf.einsum('ijk,ilk->ijl', X, Y)  # B*N*M

    return X2 + Y2 - 2 * XY  # B*N*M

def pdist(X):
    '''
    Compute Pairwise distance between X and Y
    :param X: B*N*D
    :return: D: B*N*N
    '''

    X2 = tf.expand_dims(tf.reduce_sum(X ** 2, axis=-1), axis=-1)  # B*N*1
    Y2 = tf.einsum('ijk->ikj',X2)  # B*1*N
    XY = tf.einsum('ijk,ilk->ijl', X, X)  # B*N*N

    return tf.sqrt(tf.maximum(X2 + Y2 - 2 * XY,0))  # B*N*N

def pdist_np(X):
    '''
    Compute Pairwise distance between X and X
    :param X: N*D
    :return: D: N*N
    '''

    X2 = np.sum(X ** 2, axis=-1,keepdims=True)  # N*1
    # Y2 = np.einsum('jk->kj',X2)  # 1*N
    # XY = np.einsum('jk,lk->jl', X, X)  # N*N
    D = X2 + np.einsum('jk->kj',X2) - 2 * np.einsum('jk,lk->jl', X, X)
    D[D<0.] = 0.

    return np.sqrt(D)  # N*N

def batch_gather_v1(X, idx):
    '''
    batch gather function
    :param X: Input tensor to be sliced/gathered B*N*D
    :param idx: Slicing/Gather index B*N*Knn
    :return: Xgather: Sliced/Gathered X B*N*Knn*D
    '''

    def body(X, idx, Xall, i):
        Xtmp = tf.gather(X[i, ...], idx[i, ...], 0)
        Xall = tf.concat([Xall, tf.expand_dims(Xtmp, 0)], axis=0)
        i += 1
        return X, idx, Xall, i

    B_tf = tf.shape(X)[0]

    # B = X.get_shape()[0]
    D = X.get_shape()[-1].value
    knn = idx.get_shape()[-1].value
    N_tf = tf.shape(X)[1]
    # D_tf = tf.shape(X)[2]
    dims = tf.stack([0, N_tf, knn, D])
    Xall = tf.fill(dims=dims, value=0.)
    Xall_shp = tf.TensorShape([None, Xall.get_shape()[1], Xall.get_shape()[2], Xall.get_shape()[3]])

    i = tf.constant(0)

    condition = lambda X, idx, Xall, i: tf.less(i, B_tf)

    X, idx, Xall, i = tf.while_loop(condition, body, [X, idx, Xall, i],
                                    shape_invariants=[X.get_shape(), idx.get_shape(), Xall_shp, i.get_shape()])

    return Xall

def pdist2_L2(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pdist2_L2'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))

    return D

def pdist_L2_batch(X):
    "pairwise distance for input tensor"
    #
    #   X ~ B*N*D

    N = tf.shape(X)[1]
    X_2 = tf.tile(tf.reduce_sum(X ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X, X)
    Dmat = tf.sqrt(X_2 + Y_2 - 2 * XY)  # B*N*N

    return Dmat

def IoU(pred,gt,K):
    '''
    function to measure IoU for batch input
    :param pred: B*N
    :param gt: B*N
    :return:
    '''
    
    B = gt.shape[0]
    N = gt.shape[1]

    # start_time = time.time()
    pred_onehot = np.zeros(shape=[B,N,K],dtype=int)
    gt_onehot = np.zeros(shape=[B,N,K],dtype=int)
    
    for b in range(B):
        for k in range(K):
            pred_onehot[b,pred[b]==k,k] += 1
            gt_onehot[b,gt[b]==k,k] += 1
    
    intersect = np.sum(pred_onehot * gt_onehot, axis=1)
    union = np.sum(pred_onehot + gt_onehot, axis=1) - intersect
    iou = intersect / (union+1e-6)

    # end_time = time.time()
    # print('time {}'.format(end_time - start_time))

    return iou  # B*K

def IoU_detail(pred, gt, K):
    '''
    function to measure IoU for batch input
    :param pred: B*N
    :param gt: B*N
    :return:
    '''

    B = gt.shape[0]
    N = gt.shape[1]

    # start_time = time.time()
    pred_onehot = np.zeros(shape=[B, N, K], dtype=int)
    gt_onehot = np.zeros(shape=[B, N, K], dtype=int)

    for b in range(B):
        for k in range(K):
            pred_onehot[b, pred[b] == k, k] += 1
            gt_onehot[b, gt[b] == k, k] += 1

    intersect = np.sum(pred_onehot * gt_onehot, axis=1)
    union = np.sum(pred_onehot + gt_onehot, axis=1) - intersect
    iou = intersect / (union + 1e-6)

    return iou, intersect, union  # B*K

def L2NormVec(x):
    '''
    L2 normalize vector
    :param x: vector to be normalized N
    :return:
    '''

    return np.sqrt(x/np.sum(x**2))

def L1NormVec(x):
    '''
    L1 normalize vector
    :param x: vector to be normalized N
    :return:
    '''

    return x/np.sum(np.abs(x))

def printout(str,write_flag = False, fid=None,end=''):
    '''
    function to print the string (str) and write into a file if fid is provided
    :param str:
    :param fid:
    :return:
    '''

    if write_flag:

        print(str,end=end)
        fid.write(str+end)
    else:

        print(str,end=end)

def batch_norm(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed

def ResamplePointCloud(X,target_num_pts):
    '''
    function to resample points from given point cloud X
    :param X:   Given point cloud N*D
    :param target_num_pts:  Target number of points to be resampled
    :return:
    '''

    N = X.shape[0]
    if N > target_num_pts:
        idx = np.arange(0,N)
        samp_idx = np.random.choice(idx,target_num_pts,False)
        return X[samp_idx,:], samp_idx
    elif N < target_num_pts:
        idx = np.arange(0,N)
        samp_idx = np.random.choice(idx,target_num_pts,True)
        return X[samp_idx,:], samp_idx
    else:
        samp_idx = np.arange(0,N)
        return X, samp_idx

class TF_Computation():

    def __init__(self):

        pass

    class InnerProd():

        def __init__(self):

            self.X_ph = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='X')    # B*N*D
            self.Y_ph = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='Y')
            self.Z = tf.einsum('ijk,ilk->ijl',self.X_ph, self.Y_ph)

        def Eval(self,sess,X,Y):

            return sess.run(self.Z,feed_dict={self.X_ph:X,self.Y_ph:Y})

    class PairDist2():

        def __init__(self):

            self.X_ph  = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='X')
            # self.Y_ph  = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='Y')

            N = tf.shape(self.X_ph)[1]
            X_2 = tf.tile(tf.reduce_sum(self.X_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.X_ph, self.X_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat<0,tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat>0,tf.float32) * Dmat
            self.Dmat = Dmat

        def Eval(self,sess,X):

            return sess.run(self.Dmat,feed_dict={self.X_ph:X})

    class PairWeight2():

        def __init__(self):

            self.X_ph  = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='X')
            self.gamma  = tf.placeholder(shape=(),dtype=tf.float32,name='gamma')

            N = tf.shape(self.X_ph)[1]
            X_2 = tf.tile(tf.reduce_sum(self.X_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.X_ph, self.X_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat<0,tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat>0,tf.float32) * Dmat
            Wmat = tf.exp(-Dmat/self.gamma)

            self.Wmat = Wmat

        def Eval(self,sess,X,gamma):

            return sess.run(self.Wmat,feed_dict={self.X_ph:X, self.gamma:gamma})

    class LaplacianMat():

        def __init__(self):

            self.W_ph = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='WeightMat')
            self.D = tf.matrix_diag(tf.reduce_sum(self.W_ph,axis=-1))
            self.Lmat = self.D - self.W_ph
            # self.Lsymmat = self.D**-0.5 @ self.Lmat @ self.D**-0.5

        def Eval(self,sess,W):

            return sess.run(self.Lmat,feed_dict={self.W_ph:W})

    class LaplacianMatSym():

        def __init__(self):

            self.W_ph = tf.placeholder(shape=[None,None,None],dtype=tf.float32,name='WeightMat')
            d = tf.reduce_sum(self.W_ph,axis=-1)
            D = tf.matrix_diag(d+1e-8)
            D_negsqrt = tf.matrix_diag(d**-0.5)
            Lmat = D - self.W_ph
            self.Lsymmat = D_negsqrt @ Lmat @ D_negsqrt

        def Eval(self,sess,W):

            return sess.run(self.Lsymmat,feed_dict={self.W_ph:W})

    class LaplacianMatSym_DirectComp():

        def __init__(self):

            self.X_ph = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='X')

            N = tf.shape(self.X_ph)[1]
            X_2 = tf.tile(tf.reduce_sum(self.X_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.X_ph, self.X_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat

            Wmat = tf.exp(-Dmat * 1e3)

            d = tf.reduce_sum(Wmat,axis=-1)
            D = tf.matrix_diag(d+1e-8)
            D_negsqrt = tf.matrix_diag(d**-0.5)
            Lmat = D - Wmat
            self.Lsymmat = D_negsqrt @ Lmat @ D_negsqrt

        def Eval(self,sess,X):

            return sess.run(self.Lsymmat,feed_dict={self.X_ph:X})

    class LaplacianMat_XYZRGB_DirectComp():

        def __init__(self):

            self.X_ph = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='X')
            self.RGB_ph = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='RGB')

            N = tf.shape(self.X_ph)[1]
            #### XYZ Weight Matrix
            X_2 = tf.tile(tf.reduce_sum(self.X_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.X_ph, self.X_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat
            W_XYZ = tf.exp(-Dmat * 1e3)

            #### RGB Weight Matrix
            X_2 = tf.tile(tf.reduce_sum(self.RGB_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.RGB_ph, self.RGB_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat
            W_RGB = tf.exp(-Dmat * 1e1)

            Wmat = W_XYZ * W_RGB

            d = tf.reduce_sum(Wmat,axis=-1)
            D = tf.matrix_diag(d+1e-8)
            self.Lmat = D - Wmat

        def Eval(self, sess, X, RGB):
            return sess.run(self.Lmat, feed_dict={self.X_ph: X ,self.RGB_ph: RGB})

    class LaplacianMatSym_XYZRGB_DirectComp():

        def __init__(self):

            self.X_ph = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='X')
            self.RGB_ph = tf.placeholder(shape=[None, None, None], dtype=tf.float32, name='RGB')

            N = tf.shape(self.X_ph)[1]
            #### XYZ Weight Matrix
            X_2 = tf.tile(tf.reduce_sum(self.X_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.X_ph, self.X_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat
            W_XYZ = tf.exp(-Dmat * 1e3)

            #### RGB Weight Matrix
            X_2 = tf.tile(tf.reduce_sum(self.RGB_ph ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
            Y_2 = tf.einsum('ijk->ikj', X_2)
            XY = tf.einsum('ijk,ilk->ijl', self.RGB_ph, self.RGB_ph)
            Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
            Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat
            W_RGB = tf.exp(-Dmat * 1e1)

            Wmat = W_XYZ * W_RGB

            d = tf.reduce_sum(Wmat,axis=-1)
            D = tf.matrix_diag(d+1e-8)
            D_negsqrt = tf.matrix_diag(d**-0.5)
            Lmat = D - Wmat
            self.Lsymmat = D_negsqrt @ Lmat @ D_negsqrt

        def Eval(self, sess, X, RGB):
            return sess.run(self.Lsymmat, feed_dict={self.X_ph: X, self.RGB_ph: RGB})