import tensorflow as tf

class LabelPropagation_TF():
    '''
    The baseline method for label propagation. The closed-form solution is adopted for label propagation.
    '''

    def __init__(self, alpha, beta, K):
        self.alpha = alpha
        self.beta = beta
        # self.K = K
        self.G_ph = tf.placeholder(shape=[None, None], dtype=tf.float32)  # Input predictions N*K
        self.L_ph = tf.placeholder(shape=[None, None], dtype=tf.float32)  # input laplacian matrix N*N
        self.alpha_ph = tf.placeholder(shape=(), dtype=tf.float32)  # input alpha
        self.beta_ph = tf.placeholder(shape=(), dtype=tf.float32)  # input beta

        N = tf.shape(self.G_ph)[0]

        self.w = self.ComputeWeight4EachPoint()  # weight for each point N

        self.Y = self.beta_ph * tf.linalg.inv(
            self.alpha_ph * self.L_ph + self.beta_ph * tf.diag(self.w) + 1e-5 * tf.eye(N)) @ tf.diag(self.w) @ self.G_ph
        self.Y_prob = self.Y / tf.reduce_sum(self.Y, axis=-1, keepdims=True)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta(self, beta):
        self.beta = beta

    def ComputeWeight4EachPoint(self):
        '''
        Compute weight for each sample
        :param G:   The prediction  N*K
        :return:
        '''

        K = tf.cast(tf.shape(self.G_ph)[-1], tf.float32)
        w = 1. - (-tf.reduce_sum(self.G_ph * tf.math.log(self.G_ph + 1e-5) / tf.math.log(2.), axis=1)) / (
                    tf.math.log(K) / tf.math.log(2.))  # N

        return w

    def SolveLabelProp(self, sess, L, G):
        '''
        Solve label propagation with closed-form solution
        :param L:   Laplacian matrix N*N (sparse)
        :param G:   network prediction N*K (dense)
        :return:
        '''

        self.Y_val, self.Y_prob_val, self.w_val = sess.run([self.Y, self.Y_prob, self.w], feed_dict={self.G_ph: G,
                                                                                                     self.L_ph: L,
                                                                                                     self.alpha_ph: self.alpha,
                                                                                                     self.beta_ph: self.beta})

        return self.Y_val, self.Y_prob_val, self.w_val

    def EvalWeight4EachPoint(self, sess, G):
        self.w_val = sess.run([self.w], feed_dict={self.G_ph: G})

        return self.w_val

