# Please kindly note that the neural network architecture referred to the open source project
# contributed by Michaël Defferrard et al. (https://github.com/mdeff/cnn_graph).
# However, the input in the original project is limited to a fixed graph structure,
# i.e. the graph structures of input samples are the same; 
# in this modified version, the graph structures of input samples are different.

# Copyright (c) 2016 Michaël Defferrard


import tensorflow as tf
import numpy as np
import os, time, collections, shutil

import sklearn, sklearn.datasets, sklearn.utils
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import re

class gcnn(object):
    """
    GCNN model. 
    input:
        Vs: Dimensions of vectice vecter (Channes).
        As: Number of vertices.          (Pixels)

    hyper-parameters:
        Fs: Number of feature maps.
        Ks: List of polynomial orders, i.e. filter sizes or number of hopes.
        Ps: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        Cs: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation.
        brelu:  bias and relu.
        pool:   pooling.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, Vs, As, Fs, Ks, Ps, Cs, filter='monomial', brelu='b1relu', pool='mpool1',
                num_epochs=20, batch_size=100, decay_steps=None, eval_frequency=200,
                regularization=0, dropout=0, learning_rate=0.1, decay_rate=0.95, momentum=0.9,
                dir_name=''):
        super().__init__()
        # Verify the consistency w.r.t. the number of layers.
        assert len(Fs) == len(Ks) == len(Ps)
        assert np.all(np.array(Ps) >= 1)
        p_log2 = np.where(np.array(Ps) > 1, np.log2(Ps), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        
        # Print information about NN architecture.
        Ngconv = len(Ps)
        Nfc = len(Cs)
        is_print = True
        if is_print:
            print('GCNN architecture:')
            print(' Number of graph convolution layers = {}'.format(Ngconv))
            print(' Number of fully connection layers  = {}'.format(Nfc))
            print('  input: {0}x{0} Graph with {1} dimensionaal vecter.'.format(As, Vs))
            for i in range(Ngconv):
                print('  layer {0}: cgconv{0}'.format(i+1))
                print('    representation: A_{0} * F_{1} / P_{1} = {2} * {3} / {4} = {5}'.format(
                        i, i+1, As, Fs[i], Ps[i], As*Fs[i]//Ps[i]))
                F_last = Fs[i-1] if i > 0 else Vs
                print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                        i, i+1, F_last, Fs[i], Ks[i], F_last*Fs[i]*Ks[i]))
                print('    biases: F_{} = {}'.format(i+1, Fs[i]))
            for i in range(Nfc):
                name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
                print('  layer {}: {}'.format(Ngconv+i+1, name))
                print('    representation: M_{} = {}'.format(Ngconv+i+1, Cs[i]))
                C_last = Cs[i-1] if i > 0 else As if Ngconv == 0 else As * Fs[-1] // Ps[-1]
                print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                        Ngconv+i, Ngconv+i+1, C_last, Cs[i], C_last*Cs[i]))
                print('    biases: M_{} = {}'.format(Ngconv+i+1, Cs[i]))

        # Store attributes and bind operations.        
        self.regularizers = []
        self.Vs, self.As, self.Fs, self.Ks, self.Ps, self.Cs = Vs, As, Fs, Ks, Ps, Cs
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        
        # Build the computational graph.
        self.build_graph(Vs, As)
    
    # High-level interface which runs the constructed computational graph.
    def predict(self, vertices, adjacencies, labels, sess=None):
        loss = 0
        size = vertices.shape[0]
        predictions = np.empty(size)
        probabilities = np.empty((size, 2))
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_vertices    = np.zeros((self.batch_size, vertices.shape[1], vertices.shape[2]))
            batch_adjacencies = np.zeros((self.batch_size, adjacencies.shape[1], adjacencies.shape[2]))
            batch_labels      = np.zeros(self.batch_size)

            batch_vertices[:end-begin]    = vertices[begin:end,...]
            batch_adjacencies[:end-begin] = adjacencies[begin:end,...]
            batch_labels[:end-begin]      = labels[begin:end]

            feed_dict = {self.ph_vertices:    batch_vertices,
                         self.ph_adjacencies: batch_adjacencies,
                         self.ph_labels:      batch_labels,
                         self.ph_dropout:     1}

            # batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
            # loss += batch_loss
            
            # predictions[begin:end] = batch_pred[:end-begin]

            batch_logits, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
            loss += batch_loss

            probabilities[begin:end] = batch_logits[0][:end-begin]
            predictions[begin:end] = batch_logits[1][:end-begin]
        
        probabilities = np.column_stack((probabilities, predictions, labels))
        return probabilities, predictions, loss * self.batch_size / size
        
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    def create_input_variable(self, input):
        for i in range(len(input)):
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        return input
    # Methods to construct the computational graph.
    
    def build_graph(self, Vs, As):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_vertices = tf.placeholder(tf.float32, (self.batch_size, As, Vs), 'vertices')
                self.ph_adjacencies = tf.placeholder(tf.float32, (self.batch_size, As, As), 'adjacencies')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_vertices, self.ph_adjacencies, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss,
                                          self.learning_rate,
                                          self.decay_steps,
                                          self.decay_rate,
                                          self.momentum)
            # self.op_prediction = self.prediction(op_logits)            # for output the probabilities. 06.29
            self.op_prediction = self.logitsvalue(op_logits)
            if (self.batch_size == 1):
                self.op_prediction = self.logitsvalue(op_logits)         # for predicting one sample.
             

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()
    
    def inference(self, vertices, adjacencies, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        vertices: size N x M x 
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        # Graph convolutional layers.
        # vertices     N_ * M_ * F_
        # adjacencies  M_ * M_
        N_, M_, F_, = vertices.get_shape()
        # print("x_dim_one: {0}, {1}, {2}".format(N_, M_, F_))
        for i in range(len(self.Ps)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    vertices = self.filter(vertices, adjacencies, self.Fs[i], self.Ks[i])
                with tf.name_scope('bias_relu'):
                    vertices = self.brelu(vertices)
                with tf.name_scope('pooling'):
                    vertices = self.pool(vertices, self.Ps[i])
        
        # Fully connected hidden layers.
        N, M, F = vertices.get_shape()
        # print("x_dim_two: {0}, {1}, {2}".format(N, M, F))
        vertices = tf.reshape(vertices, [int(N), int(M*F)])  # N x (M*F)
        for i, M in enumerate(self.Cs[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                vertices = self.fc(vertices, M)
                vertices = tf.nn.dropout(vertices, dropout)
        
        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            vertices = self.fc(vertices, self.Cs[-1], relu=False)
        return vertices
    

    def monomial(self, vertices, adjacencies, Fout, K):
        N, M, Fin = vertices.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        assert N == adjacencies.shape[0]
        assert M == adjacencies.shape[1] == adjacencies.shape[2]
        W = self._weight_variable([Fin*K, Fout], regularization=True)    # Fin*K X Fout # False
        X=[]
        # The efficiency can be improved by means of vectorization.
        for i in range(0, N):
            Li = adjacencies[i]                                   # M x M
            # Transform to monomial basis
            Xi_0 = vertices[i]                                    # M x Fin
            Xi   = tf.expand_dims(Xi_0, 0)                        # 1 x M x Fin
            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)                        # 1 x M x Fin
                return tf.concat([x, x_], axis=0)                 # K x M x Fin    
            for k in range(1, K):
                Xi_0 = tf.matmul(Li, Xi_0)    # M x Fin*N
                Xi   = concat(Xi, Xi_0)
            Xi = tf.reshape(Xi, [K, M, Fin])                      # K x M x Fin
            Xi = tf.transpose(Xi, [1, 2, 0])                      # M x Fin x K
            Xi = tf.reshape(Xi, [M, Fin*K])                       # M x Fin*K
            Xi = tf.matmul(Xi, W)                                 # [M x Fin*K] x[Fin*k x Fout] = [M X Fout]
            X.append(Xi)
        return tf.reshape(X, [N, M, Fout])

    def b1relu(self, vertices):
        """Bias and ReLU. One bias per filter."""
        N, M, F = vertices.get_shape()
        b       = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(vertices + b)

    def m1pool(self, vertices, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            # remain to be improved.
            return vertices
        else:
            return vertices

    def fc(self, vertices, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min   = vertices.get_shape()
        W        = self._weight_variable([int(Min), Mout], regularization=True)
        b        = self._bias_variable([Mout], regularization=False)
        vertices = tf.matmul(vertices, W) + b
        return tf.nn.relu(vertices) if relu else vertices

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def logitsvalue(self, logits):
        """Return the logits values."""
        with tf.name_scope('prediction'):
            probabilities = tf.nn.softmax(logits)
            prediction = tf.argmax(logits, axis=1)
            return probabilities, prediction
            
    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step    = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum   == 0:
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads         = optimizer.compute_gradients(loss)
            op_gradients  = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            print(self._get_path('checkpoints'))
            print(filename)
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def fit(self, train_vertices, train_adjacencies, train_labels, val_vertices, val_adjacencies, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        accuracies, accuracies_refined, losses = [], [], []
        n_train = train_vertices.shape[0]
        indices = collections.deque()
        num_steps = int(self.num_epochs * n_train / self.batch_size)
        for step in range(1, num_steps+1):
            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(n_train))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_vertices, batch_adjacencies, batch_labels = train_vertices[idx,:], train_adjacencies[idx,:], train_labels[idx]

            if type(batch_vertices) is not np.ndarray:
                batch_vertices = batch_vertices.toarray()  # convert sparse matrices
            if type(batch_adjacencies) is not np.ndarray:
                batch_adjacencies = batch_adjacencies.toarray()  # convert sparse matrices
            feed_dict = {self.ph_vertices:    batch_vertices,
                         self.ph_adjacencies: batch_adjacencies,
                         self.ph_labels:      batch_labels,
                         self.ph_dropout:     self.dropout}

            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / n_train
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                
                string, accuracy, accuracy_refined, f1, loss, probabilities = self.evaluate(val_vertices, val_adjacencies, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                accuracies_refined.append(accuracy_refined)

                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return accuracies, accuracies_refined, losses, t_step

    def predictOne(self, vertices, adjacencies, labels, inFIDs=None, sess=None):
        loss = 0
        size = vertices.shape[0]
        predictions = np.empty(size)
        probabilities = np.empty((size, 2))
        sess = self._get_session(sess)
        #print('begin...')
        #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))
        #print(tf.local_variables())
        #for var in tf.global_variables():
        #    print(var.name)
        #print('end')
        for i in range(0, size):
            one_vertices    = np.zeros((1, vertices.shape[1], vertices.shape[2]))
            one_adjacencies = np.zeros((1, adjacencies.shape[1], adjacencies.shape[2]))
            one_labels      = np.zeros(1)
            
            one_vertices[:1]    = vertices[i,...]
            one_adjacencies[:1] = adjacencies[i,...]
            one_labels[:1]      = labels[i]

            feed_dict = {self.ph_vertices:    one_vertices,
                         self.ph_adjacencies: one_adjacencies,
                         self.ph_labels:      one_labels,
                         self.ph_dropout:     1}

            one_logits, one_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
            loss += one_loss

            probabilities[i] = one_logits[0][0]
            predictions[i] = one_logits[1]
            #if predictions[i] != labels[i]:
                # print(inFIDs[i])
                # print('{:^4}, {}, {:.5f}, {:.5f}, {:1.5f}'.format(inFIDs[i], labels[i], probabilities[0], probabilities[1], one_loss))
            print('{:^4},  {:.4f},  {:.4f}'.format(inFIDs[i], probabilities[i][0], probabilities[i][1]))

        probabilities = np.column_stack((probabilities, predictions, labels))
        return probabilities, predictions, loss / size

    def evaluate(self, vertices, adjacencies, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        probabilities, predictions, loss = self.predict(vertices, adjacencies, labels, sess)
        #print(predictions)

        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')

        ncorrects_refined = 0; nshape = probabilities.shape
        for i in range(0, nshape[0]):
            if probabilities[i][nshape[1]-2] == probabilities[i][nshape[1]-1] and probabilities[i][labels[i]] > 0.8:
                ncorrects_refined = ncorrects_refined + 1
        accuracy_refined = 100 * ncorrects_refined / nshape[0]

        # string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
        #         accuracy, ncorrects, len(labels), f1, loss)

        string = 'accuracy: {:.2f} ({:d} / {:d}), accuracy_refined: {:.2f} ({:d} / {:d}), loss: {:.2e}'.format(accuracy, ncorrects, len(labels), accuracy_refined, ncorrects_refined, len(labels), loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)

        return string, accuracy, accuracy_refined, f1, loss, probabilities

    def evaluateOne(self, vertices, adjacencies, labels, inFIDs, sess=None):
        t_process, t_wall = time.process_time(), time.time()
        probabilities, predictions, loss = self.predictOne(vertices, adjacencies, labels, inFIDs, sess)

        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)

        string = 'accuracy: {:.2f} ({:d} / {:d}), loss: {:.2e}'.format(accuracy, ncorrects, len(labels), loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, accuracy, loss, probabilities


class model_perf(object):

    def __init__(s):
        s.names, s.params = set(), {}
        s.fit_accuracies, s.fit_losses, s.fit_time = {}, {}, {}
        s.train_accuracy, s.train_f1, s.train_loss = {}, {}, {}
        s.test_accuracy, s.test_f1, s.test_loss = {}, {}, {}
        s.fit_accuracies_refined, s.train_accuracy_refined, s.test_accuracy_refined = {}, {}, {}

    def testOne(s, model, test_dataset):
        string, accuracy, loss, probabilities = model.evaluateOne(test_dataset[0], test_dataset[1], test_dataset[2], test_dataset[3])

        print('test  {}'.format(string))
        print("Confusion martix = \n{}".format(sklearn.metrics.confusion_matrix(probabilities[:,-2], probabilities[:,-1])))

    def test(s, model, name, params, train_dataset, val_dataset, test_dataset):
        s.params[name] = params

        s.fit_accuracies[name], s.fit_accuracies_refined[name], s.fit_losses[name], s.fit_time[name] = \
                model.fit(train_dataset[0], train_dataset[1], train_dataset[2], val_dataset[0], val_dataset[1], val_dataset[2])

        string, s.train_accuracy[name], s.train_accuracy_refined[name], s.train_f1[name], s.train_loss[name], probabilities = \
                model.evaluate(train_dataset[0], train_dataset[1], train_dataset[2])

        print('train {}'.format(string))
        string, s.test_accuracy[name], s.test_accuracy_refined[name], s.test_f1[name], s.test_loss[name], probabilities = \
                model.evaluate(test_dataset[0], test_dataset[1], test_dataset[2])

        print('test  {}'.format(string))
        
        # file = r'C:\Users\Administrator\Desktop\ahtor\thesis\gcnn\data\_used3.txt'
        # np.savetxt(file, probabilities, fmt='%.2f')
        
        print("Confusion martix = \n{}".format(sklearn.metrics.confusion_matrix(probabilities[:,-2], probabilities[:,-1])))

        s.names.add(name)