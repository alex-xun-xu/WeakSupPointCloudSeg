import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent


def focal_loss_v1(prediction_tensor, target_tensor, alpha=None, weights=None, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """

    if alpha is None:
        alpha = 0.25*tf.ones_like(prediction_tensor)

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent



def class_weighted_CE_loss(pred, gt,posWeight,negWeight):
    '''
    Weighted Cross Entropy loss
    :param pred: (unnormalized) prediction tensor B*1*K
    :param gt: ground-truth tensor B*1*K
    :param weight: weight for each class K
    :return:
    '''

    sigmoid_p = tf.nn.sigmoid(pred) # B*1*K
    loss = - (posWeight*gt * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) + negWeight*(1-gt)*tf.log(tf.clip_by_value(1-sigmoid_p, 1e-8, 1.0)))
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt,logits=pred)
    return loss

def SelfEntropy(Z):
    '''
    Self Entropy Loss to penalize uncertain predictions
    :param Z:   Logits before pooling layer B*N*D
    :return: H_Z:   Self-Entropy B*N
    '''

    Z_hat = tf.nn.softmax(Z,axis=-1)

    eps = 1e-5
    H_Z = tf.reduce_sum(Z_hat*tf.log(Z_hat+eps),axis=-1)

    return H_Z

def OverwhelmLoss_v1(L,Y):
    '''
    A loss to enforce the max logits for class j1 is higher than the min logits for class j2 with both j1 and j2 being positive classes
    :param L: The logits for each point     B*N*K
    :param Y: The GT for each point cloud     B*K
    :return:
    '''

    K = tf.shape(Y)[-1]

    L_max = tf.expand_dims(tf.reduce_max(L,axis=1),axis=-1) # B*K*1
    L_min = tf.reduce_min(L,axis=1,keepdims=True) # B*1*K

    L_max_mat = tf.tile(L_max,multiples=[1,1,K])    # B*K*K
    L_min_mat = tf.tile(L_min,multiples=[1,K,1])    # B*K*K

    OverwhelmPenalty = tf.maximum(L_min_mat - L_max_mat,tf.zeros_like(L_max_mat))   # B*K*K

    Y_mat = tf.expand_dims(Y,axis=-1)   # B*K*1
    Mask = tf.einsum('ijk,ilk->ijl',Y_mat,Y_mat) - tf.matrix_diag(Y)    # B*K*K

    loss = tf.reduce_mean(OverwhelmPenalty * Mask,axis=[-1,-2])  # B
    loss = tf.reduce_mean(loss)

    return loss


def OverwhelmLoss_v2(L,Y):
    '''
    A loss to enforce the for at least one sample, the logit for positive class j1 is higher than any other classes
    :param L: The logits for each point     B*N*K
    :param Y: The GT for each point cloud     B*K
    :return:
    '''

    K = L.get_shape()[-1].value

    #### Overwhelm Loss
    loss_full_pos = []
    loss_full_neg = []

    for k in range(K):

        L_k = tf.gather(L,k,axis=-1) #B*N
        L_exclude_k = np.arange(0,K)
        L_exclude_k = np.delete(L_exclude_k, k)
        max_Lij = tf.reduce_max(tf.gather(L,L_exclude_k,axis=-1),axis=-1)   # B*N

        ## Positive Classes
        min_i_max_L_minus_L = tf.reduce_min(max_Lij - L_k,axis=1)   # B

        max_mingap_0 = tf.maximum(min_i_max_L_minus_L, 0)  # B

        loss_full_pos.append(tf.gather(Y,k,axis=-1)*max_mingap_0)   # B

        ## Negative Classes
        max_i_L_minus_max_L = tf.reduce_max(L_k - max_Lij, axis=1)  # B

        max_maxgap_0 = tf.maximum(max_i_L_minus_max_L, 0)  # B

        loss_full_neg.append((1 - tf.gather(Y, k, axis=-1)) * max_maxgap_0)  # B


    #### All loss
    loss_full_pos = tf.stack(loss_full_pos,axis=-1) #   B*K
    loss_full_neg = tf.stack(loss_full_neg,axis=-1) #   B*K

    loss = tf.reduce_mean(loss_full_pos + loss_full_neg)


    return loss, loss_full_pos, loss_full_neg


def OverwhelmLoss(L,Y):
    '''
    A loss to enforce the for at least one sample, the logit for positive class j1 is higher than any other classes
    :param L: The logits for each point     B*N*K
    :param Y: The GT for each point cloud     B*K
    :return:
    '''

    K = tf.shape(Y)[-1]

    max_j_Lij = tf.reduce_max(L,axis=-1,keepdims=True)    # B*N*1

    min_i_max_L_minus_L = tf.reduce_min(max_j_Lij - L,axis=1)   # B*K

    max_maxgap_0 = tf.maximum(min_i_max_L_minus_L,0)    # B*K

    loss_full = Y*max_maxgap_0   # B*K

    loss = tf.reduce_sum(loss_full,axis=-1)

    loss = tf.reduce_mean(loss)

    return loss, loss_full


