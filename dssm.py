import numpy as np
import tensorflow as tf
import words_to_matrix as wtm

trigram_map = wtm.gen_trigrams()

# DSSM model

class Dssm(tf.keras.Model):
    def __init__(self):
        super(Dssm, self).__init__()
        self.limit1 = np.sqrt(6.0 / (len(trigram_map) + 1 + 300))
        self.W1 = tf.Variable(initial_value=tf.random.uniform([len(trigram_map)+1, 300], -self.limit1, self.limit1),
                             name="weight1") # weight1
        self.b1 = tf.Variable(tf.random.uniform([300], -self.limit1, self.limit1), name="bias1")  # bias1
        self.limit2 = np.sqrt(6.0 / (300 + 300))
        self.W2 = tf.Variable(initial_value=tf.random.uniform([300, 300], -self.limit2, self.limit2),
                              name="weight2") # weight2
        self.b2 = tf.Variable(tf.random.uniform([300], -self.limit2, self.limit2), name="bias2")  # bias2
        self.limit3 = np.sqrt(6.0 / (300 + 128))
        self.W3 = tf.Variable(initial_value=tf.random.uniform([300, 128], -self.limit3, self.limit3),
                              name="weight3") # weight3
        self.b3 = tf.Variable(tf.random.uniform([128], -self.limit3, self.limit3), name="bias3")  # bias3

    def layer(self, x): # linear transformation
        """
            applying 3 layers: 300, 300, 128
            return with size 128
        """
        x = tf.nn.tanh(tf.matmul(x, self.W1) + self.b1)
        x = tf.nn.tanh(tf.matmul(x, self.W2) + self.b2)
        x = tf.nn.tanh(tf.matmul(x, self.W3) + self.b3)
        return x

    def cosine_similarity(self, A, B):
        """
            scalar multiply of two vectors: title and query
        """
        Anorm = tf.nn.l2_normalize(A, axis=1)  # normalize A
        Bnorm = tf.nn.l2_normalize(B, axis=1)  # normalize B
        sim = tf.reduce_sum(Anorm * Bnorm, axis=1)  # dot product of normalized A and normalized B
        return sim

    def __call__(self, data, training=True): # train model for each pair (query, title) in dataset
        self.all_similarities = []
        for pair in data:
            query = self.layer(pair[0])
            title = self.layer(pair[1])
            similarity = self.cosine_similarity(query, title)
            self.all_similarities.append(similarity[0])
        return tf.nn.softmax(self.all_similarities) # result - softmax of all cosine_similarities