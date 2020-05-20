import tensorflow as tf
import numpy as np
import sys

sys.path.append('/home/elexuxu/vision01/WeakSup/Tool')

import Tool

def Loss_SpatialSmooth(X,W,Ind):
    '''
    function to return spatial smoothness constraint loss
    :param X: Input point cloud  float B*N*3
    :param W: Input knn weight matrix  float B*N*Knn
    :param Ind: Input knn index  int B*N*Knn
    :return: spatial smooth loss
    '''

    knn = W.get_shape()[-1].value
    # unstacked_X = tf.unstack(X,axis=0)
    # unstacked_Ind = tf.unstack(Ind,axis=0)
    #
    # X_tilde = []
    # for X_b, Ind_b in zip(unstacked_X,unstacked_Ind):
    #
    #     X_tilde.append(tf.gather(X_b,Ind_b))

    X_tilde = Tool.batch_gather_v1(X,Ind)   # indexed X B*N*Knn*3

    # X_tilde = tf.stack(X_tilde,axis=0)  # indexed X B*N*Knn*3
    X_expand  = tf.tile(X[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*3
    loss = tf.reduce_mean(W*tf.reduce_sum((X_expand - X_tilde)**2,axis=-1))

    return loss


def Loss_SpatialSmooth_SelfContain(X,gamma=1e-1,knn=5):
    '''
    function to return spatial smoothness constraint loss
    :param X: Input point cloud  float B*N*3
    :param W: Input knn weight matrix  float B*N*Knn
    :param Ind: Input knn index  int B*N*Knn
    :return: spatial smooth loss
    '''

    ## Compute Full Weight Matrix
    N = tf.shape(X)[1]
    X_2 = tf.tile(tf.reduce_sum(X ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X, X)
    Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
    Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat
    # W = tf.exp(-Dmat / gamma)    # B*N*N

    ## Compute Knn Graph
    Ind = tf.argsort(Dmat)[:,:,0:knn]
    W_knn = []
    for d_i, Ind_i in zip(tf.unstack(Dmat,axis=0), tf.unstack(Ind)):
        W_knn.append(tf.batch_gather(tf.exp(-d_i / gamma), Ind_i))
    W_knn = tf.stack(W_knn)

    X_tilde = Tool.batch_gather_v1(X,Ind)   # indexed X B*N*Knn*3

    X_expand = tf.tile(X[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*3
    loss = W_knn*tf.reduce_sum((X_expand - X_tilde)**2) # B*N*Knn
    loss = tf.reduce_sum(loss)/tf.cast(tf.shape(loss)[0]*tf.shape(loss)[1]*tf.shape(loss)[2],tf.float32)

    return loss


def Loss_SpatialColorSmooth_SelfContain(Z,X,gamma=1e-1,knn=10):
    '''
    function to return spatial and color smoothness constraint loss
    :param Z: Input point cloud feature embedding float B*N*D
    :param X: Input point cloud  float B*N*6    XYZRGB
    :param W: Input knn weight matrix  float B*N*Knn
    :param Ind: Input knn index  int B*N*Knn
    :return: spatial smooth loss
    '''

    ## Compute Full Weight Matrix
    N = tf.shape(X)[1]

    # XYZ channel
    X_2 = tf.tile(tf.reduce_sum(X[:,:,0:3] ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X[:,:,0:3], X[:,:,0:3])
    Dmat_xyz = X_2 + Y_2 - 2 * XY  # B*N*N
    Dmat_xyz = tf.cast(Dmat_xyz < 0, tf.float32) * tf.zeros_like(Dmat_xyz) + tf.cast(Dmat_xyz > 0, tf.float32) * Dmat_xyz

    # RGB channel
    X_2 = tf.tile(tf.reduce_sum(X[:,:,3:6] ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X[:,:,3:6], X[:,:,3:6])
    Dmat_rgb = X_2 + Y_2 - 2 * XY  # B*N*N
    Dmat_rgb = tf.cast(Dmat_rgb < 0, tf.float32) * tf.zeros_like(Dmat_rgb) + tf.cast(Dmat_rgb > 0, tf.float32) * Dmat_rgb

    # W = tf.exp(-Dmat / gamma)    # B*N*N

    ## Compute Knn Graph
    # xyz channel
    Ind_xyz = tf.argsort(Dmat_xyz)[:,:,0:knn]   # B*N*knn
    W_knn_xyz = []
    for d_i, Ind_i in zip(tf.unstack(Dmat_xyz,axis=0), tf.unstack(Ind_xyz)):
        W_knn_xyz.append(tf.batch_gather(tf.exp(-d_i / gamma), Ind_i))
    W_knn_xyz = tf.stack(W_knn_xyz)     # B*N*knn
    # rgb channel
    Ind_rgb = tf.argsort(Dmat_rgb)[:,:,0:knn]   # B*N*knn
    W_knn_rgb = []
    for d_i, Ind_i in zip(tf.unstack(Dmat_rgb,axis=0), tf.unstack(Ind_rgb)):
        W_knn_rgb.append(tf.batch_gather(tf.exp(-d_i / gamma), Ind_i))
    W_knn_rgb = tf.stack(W_knn_rgb)     # B*N*knn

    knn_mask = tf.cast(tf.math.equal(Ind_xyz,Ind_rgb), tf.float32)   # B*N*Knn*3

    #### Compute Loss
    # knn from xyz channels
    Z_tilde = Tool.batch_gather_v1(Z,Ind_xyz)   # indexed X B*N*Knn*D
    Z_expand = tf.tile(Z[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*D
    loss_xyz = knn_mask*W_knn_xyz*tf.reduce_sum((Z_expand - Z_tilde)**2,axis=-1) # B*N*Knn
    # knn from rgb channels
    Z_tilde = Tool.batch_gather_v1(Z,Ind_rgb)   # indexed X B*N*Knn*D
    Z_expand = tf.tile(Z[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*D
    loss_rgb = knn_mask*W_knn_rgb*tf.reduce_sum((Z_expand - Z_tilde)**2,axis=-1) # B*N*Knn

    loss = loss_xyz + loss_rgb
    loss = tf.reduce_sum(loss)/tf.cast(tf.shape(loss)[0]*tf.shape(loss)[1]*tf.shape(loss)[2],tf.float32)

    return loss

def Loss_SpatialColorSmooth_add_SelfContain(Z,X,gamma=1e-1,knn=10):
    '''
    function to return spatial and color smoothness constraint loss
    :param Z: Input point cloud feature embedding float B*N*D
    :param X: Input point cloud  float B*N*6    XYZRGB
    :param W: Input knn weight matrix  float B*N*Knn
    :param Ind: Input knn index  int B*N*Knn
    :return: spatial smooth loss
    '''

    ## Compute Full Weight Matrix
    N = tf.shape(X)[1]

    # XYZ channel
    X_2 = tf.tile(tf.reduce_sum(X ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X, X)
    Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
    Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat

    # W = tf.exp(-Dmat / gamma)    # B*N*N

    ## Compute Knn Graph
    # Ind = tf.argsort(Dmat)[:,:,0:knn]   # B*N*knn
    _, Ind = tf.nn.top_k(-Dmat, k = knn, sorted=True) # B*N*knn
    W_knn = []
    for d_i, Ind_i in zip(tf.unstack(Dmat,axis=0), tf.unstack(Ind)):
        W_knn.append(tf.batch_gather(tf.exp(-d_i / gamma), Ind_i))
    W_knn = tf.stack(W_knn)     # B*N*knn

    #### Compute Loss
    Z_tilde = Tool.batch_gather_v1(Z,Ind)   # indexed X B*N*Knn*D
    Z_expand = tf.tile(Z[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*D
    loss = W_knn*tf.reduce_mean((Z_expand - Z_tilde)**2,axis=-1) # B*N*Knn
    # loss = tf.reduce_sum(loss)/tf.cast(tf.shape(loss)[0]*tf.shape(loss)[1]*tf.shape(loss)[2],tf.float32)
    loss = tf.reduce_mean(loss)

    return loss

def Loss_SpatialColorSmoothAdd_UnknownBatch_SelfContain(Z,X,gamma=1e-1,knn=10):
    '''
    function to return spatial and color smoothness constraint loss
    :param Z: Input point cloud feature embedding float B*N*D
    :param X: Input point cloud  float B*N*6    XYZRGB
    :param W: Input knn weight matrix  float B*N*Knn
    :param Ind: Input knn index  int B*N*Knn
    :return: spatial smooth loss
    '''

    ## Compute Full Weight Matrix
    N = tf.shape(X)[1]

    # XYZ channel
    X_2 = tf.tile(tf.reduce_sum(X ** 2, axis=-1, keepdims=True), multiples=[1, 1, N])  # B*N*N
    Y_2 = tf.einsum('ijk->ikj', X_2)
    XY = tf.einsum('ijk,ilk->ijl', X, X)
    Dmat = X_2 + Y_2 - 2 * XY  # B*N*N
    Dmat = tf.cast(Dmat < 0, tf.float32) * tf.zeros_like(Dmat) + tf.cast(Dmat > 0, tf.float32) * Dmat

    # W = tf.exp(-Dmat / gamma)    # B*N*N
    def custom_gather_v2(a, b):
        def apply_gather(x):
            return tf.gather(x[0], tf.cast(x[1], tf.int32))

        a = tf.cast(a, dtype=tf.float32)
        b = tf.cast(b, dtype=tf.float32)
        stacked = tf.stack([a, b], axis=1)
        gathered = tf.map_fn(apply_gather, stacked)
        return tf.stack(gathered, axis=0)
    
    ## Compute Knn Graph
    # Ind = tf.argsort(Dmat)[:,:,0:knn]   # B*N*knn
    negDmat_knn, Ind = tf.nn.top_k(-Dmat, k = knn, sorted=True) # B*N*knn
    W_knn = tf.exp(negDmat_knn/gamma)
    # W_knn = []
    #
    # W_knn = custom_gather_v2(tf.exp(-Dmat/gamma), Ind)
    
    # for d_i, Ind_i in zip(tf.unstack(Dmat,axis=0), tf.unstack(Ind)):
    #     W_knn.append(tf.batch_gather(tf.exp(-d_i / gamma), Ind_i))
    # W_knn = tf.stack(W_knn)     # B*N*knn

    #### Compute Loss
    Z_tilde = Tool.batch_gather_v1(Z,Ind)   # indexed X B*N*Knn*D
    Z_expand = tf.tile(Z[:,:,tf.newaxis,:],[1,1,knn,1])    # expanded X B*N*Knn*D
    loss = W_knn*tf.reduce_sum((Z_expand - Z_tilde)**2,axis=-1) # B*N*Knn
    # loss = tf.reduce_sum(loss)/tf.cast(tf.shape(loss)[0]*tf.shape(loss)[1]*tf.shape(loss)[2],tf.float32)
    loss = tf.reduce_mean(loss)

    return loss




def ComputeW(Target, X,knn):

    sess = Target[0]
    targetTensors = Target[1]

    W = targetTensors['W'].Eval(sess,X,0.1)
    return W

if __name__ == '__main__':
    X = tf.random.uniform(shape=[12, 1024, 3])
    # W = tf.random.uniform(shape=[12, 1024, 5])
    Ind = np.random.choice(np.arange(0, 1024), 12 * 1024 * 5)
    Ind = tf.constant(np.reshape(Ind, [12, 1024, 5]))

    Loss_SpatialSmooth_SelfContain(X)