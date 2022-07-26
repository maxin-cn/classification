# Copyright (c) Meituan.
# author: liujunjie10@meituan.com

from __future__ import absolute_import, division, print_function

import sys
import os
import time
import math
import tensorflow as tf
import numpy as np
# Import _linear
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops.variables import PartitionedVariable
from tensorflow.python.training.checkpoint_utils import init_from_checkpoint

if tuple(map(int, tf.__version__.split("."))) >= (1, 6, 0):
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    _linear = core_rnn_cell._linear
else:
    from tensorflow.python.ops.rnn_cell_impl import _linear

def get_variable_dynamic(tensor_name_in_ckpt, shape, ckpt_to_load_from=None, trainable=False):
    var = tf.get_variable(tensor_name_in_ckpt, shape=shape,
                            trainable=trainable, partitioner=partitioned_variables.min_max_variable_partitioner(max_partitions=10, min_slice_size=2 << 15))

    if ckpt_to_load_from is not None:
        to_restore = var
        if isinstance(to_restore, PartitionedVariable):
            to_restore = to_restore._get_variable_list()
        init_from_checkpoint(ckpt_to_load_from,
                                {tensor_name_in_ckpt: to_restore})
    return var

def get_embedding(ckpt_path, embed_matrix_shape, var_name_compress):
    input_scope = tf.get_variable_scope()
    with tf.variable_scope(input_scope,reuse=tf.AUTO_REUSE,partitioner=partitioned_variables.min_max_variable_partitioner(max_partitions=10, min_slice_size=2 << 15)):
        embedding_var = get_variable_dynamic(tensor_name_in_ckpt=var_name_compress, shape=embed_matrix_shape, ckpt_to_load_from=ckpt_path)
    return embedding_var

class EmbeddingCompressor(object):

    """
    _TAU = 1.0
    _BATCH_SIZE = 1 #64
    """

    def __init__(self, n_codebooks, n_centroids, batch_size, embed_shape,
                    ckpt_name, ckpt_path, export_name, var_name_compress, istrain=True):
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        model_path: prefix for saving or loading the parameters
        """
        self.M                  = n_codebooks
        self.K                  = n_centroids
        self._BATCH_SIZE        = batch_size
        self._TAU               = 1.0
        self.embed_matrix_shape = embed_shape
        self.ckpt_name          = ckpt_name
        self.ckpt_path          = ckpt_path
        self.istrain            = istrain
        self.var_name_compress  = var_name_compress
        self.export_name        = export_name

        if self.istrain == False:
            self._BATCH_SIZE = 1

    def _gumbel_dist(self, shape, eps=1e-20):
        U = tf.random_uniform(shape,minval=0,maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _sample_gumbel_vectors(self, logits, temperature):
        y = logits + self._gumbel_dist(tf.shape(logits))
        return tf.nn.softmax( y / temperature)

    def _gumbel_softmax(self, logits, temperature, sampling=True):
        """Compute gumbel softmax.

        Without sampling the gradient will not be computed
        """
        if sampling:
            y = self._sample_gumbel_vectors(logits, temperature)
        else:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def _encode(self, input_matrix, word_ids, embed_size):
        input_embeds = tf.nn.embedding_lookup(input_matrix, word_ids, name="input_embeds")

        M, K = self.M, self.K

        with tf.variable_scope("h"):
            h = tf.nn.tanh(_linear(input_embeds, M * K/2, True))
        with tf.variable_scope("logits"):
            logits = _linear(h, M * K, True)
            logits = tf.log(tf.nn.softplus(logits) + 1e-8)
        logits = tf.reshape(logits, [-1, M, K], name="logits")
        return input_embeds, logits

    def _decode(self, gumbel_output, codebooks):
        return tf.matmul(gumbel_output, codebooks)

    def _reconstruct(self, codes, codebooks):
        return None

    def build_export_graph(self):
        """Export the graph for exporting codes and codebooks.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        vocab_size = self.embed_matrix_shape[0]
        embed_size = self.embed_matrix_shape[1]

        # Define input variables
        input_matrix = get_embedding(self.ckpt_path, self.embed_matrix_shape, self.var_name_compress) # tf.constant(embed_matrix, name="embed_matrix")

        word_ids = tf.placeholder_with_default(
            np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")

        # Define codebooks
        codebooks = tf.get_variable("codebook", [self.M * self.K, embed_size])

        # Coding
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)  # ~ (B, M, K)
        codes = tf.cast(tf.argmax(logits, axis=2), tf.int32)  # ~ (B, M)

        # Reconstruct
        offset = tf.range(self.M, dtype="int32") * self.K
        codes_with_offset = codes + offset[None, :]

        selected_vectors = tf.gather(codebooks, codes_with_offset)  # ~ (B, M, H)
        reconstructed_embed = tf.reduce_sum(selected_vectors, axis=1)  # ~ (B, H)
        return word_ids, codes, reconstructed_embed

    def build_training_graph(self):
        """Export the training graph.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        vocab_size = self.embed_matrix_shape[0]
        embed_size = self.embed_matrix_shape[1]

        # Define input variables
        input_matrix = get_embedding(self.ckpt_path, self.embed_matrix_shape, self.var_name_compress) # tf.constant(embed_matrix, name="embed_matrix")

        tau = tf.placeholder_with_default(np.array(1.0, dtype='float32'), tuple()) - 0.1
        word_ids = tf.placeholder_with_default(
            np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")

        # Define codebooks
        codebooks = tf.get_variable("codebook", [self.M * self.K, embed_size])

        # Encoding
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)  # ~ (B, M, K)

        # Discretization
        D = self._gumbel_softmax(logits, self._TAU, sampling=True)
        gumbel_output = tf.reshape(D, [-1, self.M * self.K])  # ~ (B, M * K)
        maxp = tf.reduce_mean(tf.reduce_max(D, axis=2))

        # Decoding
        y_hat = self._decode(gumbel_output, codebooks)

        # Define loss
        loss = 0.5 * tf.reduce_sum((y_hat - input_embeds)**2, axis=1)
        loss = tf.reduce_mean(loss, name="loss")

        # Define optimization
        max_grad_norm = 0.001
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        global_norm = tf.identity(global_norm, name="global_norm")
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")

        return word_ids, loss, train_op, maxp

    def train(self, max_epochs=300):
        """Train the model for compress `embed_matrix` and save to `model_path`.

        Args:
            embed_matrix: a numpy matrix
        """
        vocab_size = self.embed_matrix_shape[0]
        valid_ids = np.random.RandomState(3).randint(0, vocab_size, size=(self._BATCH_SIZE * 10,)).tolist()
        # Training
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
                word_ids_var, loss_op, train_op, maxp_op = self.build_training_graph()

            # Initialize variables
            tf.global_variables_initializer().run()

            best_loss = 100000
            saver = tf.train.Saver()

            vocab_list = list(range(vocab_size))
            for epoch in range(max_epochs):
                start_time = time.time()
                train_loss_list = []
                train_maxp_list = []
                np.random.shuffle(vocab_list)
                for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                    word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                    loss, _, maxp = sess.run(
                        [loss_op, train_op, maxp_op],
                        {word_ids_var: word_ids}
                    )
                    train_loss_list.append(loss)
                    train_maxp_list.append(maxp)
                #    print(loss, maxp)
                #break
                # Print every epoch
                time_elapsed = time.time() - start_time
                bps = len(train_loss_list) / time_elapsed

                # Validation
                valid_loss_list = []
                valid_maxp_list = []
                for start_idx in range(0, len(valid_ids), self._BATCH_SIZE):
                    word_ids = valid_ids[start_idx:start_idx + self._BATCH_SIZE]
                    loss, maxp = sess.run(
                        [loss_op, maxp_op],
                        {word_ids_var: word_ids}
                    )
                    valid_loss_list.append(loss)
                    valid_maxp_list.append(maxp)

                # Report
                valid_loss = np.mean(valid_loss_list)
                report_token = ""
                if valid_loss <= best_loss * 0.999:
                    report_token = "*"
                    best_loss = valid_loss
                    saver.save(sess, os.path.join("./results", self.export_name) + ".ckpt")
                print("[epoch{}] trian_loss={:.6f} train_maxp={:.6f} valid_loss={:.6f} valid_maxp={:.6f} bps={:.0f} {}".format(
                    epoch,
                    np.mean(train_loss_list), np.mean(train_maxp_list),
                    np.mean(valid_loss_list), np.mean(valid_maxp_list),
                    len(train_loss_list) / time_elapsed,
                    report_token
                ))
        print("Training Done")

    def export(self, prefix):
        """Export word codes and codebook for given embedding.

        Args:
            embed_matrix: original embedding
            prefix: prefix of saving path
        """
        import_path = os.path.join("./results", self.export_name) + ".ckpt"
        assert os.path.exists(os.path.join("./results"))
        vocab_size = self.embed_matrix_shape[0]

        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph"):
                word_ids_var, codes_op, reconstruct_op = self.build_export_graph()
            saver = tf.train.Saver()
            saver.restore(sess, import_path)

            # Dump codebook
            codebook_tensor = sess.graph.get_tensor_by_name('Graph/codebook:0')
            np.save(prefix + ".codebook", sess.run(codebook_tensor))

            # Dump codes
            print(self._BATCH_SIZE)
            codes_list = []
            with open(prefix + ".codes", "w") as fout:
                vocab_list = list(range(self.embed_matrix_shape[0]))
                for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                    word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                    codes = sess.run(codes_op, {word_ids_var: word_ids}).tolist()
                    code_list = []
                    for code in codes:
                        code_list.append(code)
                        fout.write(" ".join(map(str, code)) + "\n")
                    codes_list.append(code_list)
            codes_arr = np.array(codes_list).astype('int8')
            np.save("dict.codes.npy", codes_arr)

    """
    def evaluate(self):
        assert os.path.exists(self._model_path + ".meta")
        vocab_size = embed_matrix_shape[0]

        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph"):
                word_ids_var, codes_op, reconstruct_op = self.build_export_graph(embed_matrix)
            saver = tf.train.Saver()
            saver.restore(sess, self._model_path)


            vocab_list = list(range(embed_matrix_shape[0]))
            distances = []
            for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                reconstructed_vecs = sess.run(reconstruct_op, {word_ids_var: word_ids})
                original_vecs = embed_matrix[start_idx:start_idx + self._BATCH_SIZE]
                distances.extend(np.linalg.norm(reconstructed_vecs - original_vecs, axis=1).tolist())
            return np.mean(distances)
    """
