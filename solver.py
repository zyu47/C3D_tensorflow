import tensorflow as tf
import numpy as np
import random
#from data_preprocessing.c3d_preprocessing import crop
from c3d_preprocessing import *

class Solver:
    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data with the following:
          'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
          'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
          'y_train': Array of shape (N_train,) giving labels for training images
          'y_val': Array of shape (N_val,) giving labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.ind_dict_train = data['index_dict_train']
        self.ind_dict_val = data['index_dict_val']


        # Unpack keyword arguments
        #self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.lr = kwargs.pop('learning_rate', 1e-2)
        #self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_classes = kwargs.pop('num_classes', 101)

        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 200)
        self.verbose = kwargs.pop('verbose', True)
        self.keep_prob = kwargs.pop('dropout', .5)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)


        sess = tf.Session()
        self.y_train = sess.run(tf.one_hot(self.y_train, self.num_classes))
        self.y_val = sess.run(tf.one_hot(self.y_val, self.num_classes))
        print 'shape of y_train and y_val ', self.y_train.shape, self.y_val.shape

        self.create_test_clips()

        self.model.saver.restore(self.model.sess, self.model.logs_path)

    def _create_batch(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """

        #Pick random 30 classes and one file from each class, process and take one random clip from that video


        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask, :,:,:,:]
        y_batch = self.y_train[batch_mask, :]

    #-------------------------------------#
        #Including crop function here
        h, w = X_batch[0][1], X_batch[0][2]
        size4crop= (112,112)

        for ind, clip in enumerate(X_batch):
            topleft_h = random.randint(0, h - size4crop[0] - 1)
            topleft_w = random.randint(0, w - size4crop[1] - 1)

            clip = [crop(img, (topleft_h, topleft_w), size4crop) for img in clip]
            X_batch[ind] = clip

    #--------------------------------------#

        print 'shape of xbatch and ybatch: ', X_batch.shape, y_batch.shape
        print y_batch
        return X_batch, y_batch

        # # Compute loss and gradient
        # loss, grads = self.model.loss(X_batch, y_batch)
        # self.loss_history.append(loss)
        #
        # # Perform a parameter update
        # for p, w in self.model.params.iteritems():
        #     dw = grads[p]
        #     config = self.optim_configs[p]
        #     next_w, next_config = self.update_rule(w, dw, config)
        #     self.model.params[p] = next_w
        #     self.optim_configs[p] = next_config
    def _create_batch_vid(self):
        # pick 30 random classes
        #classes = random.sample(range(self.num_classes), self.batch_size)
        X_batch = []
        y_batch = []
        while True:
            # randomly pick one video
            c = random.choice(range(self.num_classes))
            vid_path_ind = random.choice(self.ind_dict_train[c])
            clip = proc_video(self.X_train[vid_path_ind])
            if clip !=  None:
                X_batch.append(clip)
                y_batch.append(self.y_train[vid_path_ind])
                if len(X_batch) == self.batch_size:
                    return np.array(X_batch), np.array(y_batch)


    def create_test_clips(self):
        self.X_val_data = []
        self.y_val_data = []

        for c in self.ind_dict_val:
            vid_path_ind = self.ind_dict_val[c][:]
            for p in vid_path_ind:
                processed = proc_video(self.X_val[p], return_all=True, test_flag=True)
                self.X_val_data += processed
                self.y_val_data += [self.y_val[p] for i in range(len(processed))]
        self.X_val_data = np.array(self.X_val_data)
        self.y_val_data = np.array(self.y_val_data)

        print 'shape of X_val_data, y_val_data is: ', self.X_val_data.shape, self.y_val_data.shape


    def run(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch *100

        self.valid_summary = tf.Summary()

        for t in xrange(num_iterations):
            xbatch, ybatch = self._create_batch_vid()

            if ((t*1.0/iterations_per_epoch) % 400) == 0:
                self.lr /= self.lr_decay
            self._train(xbatch, ybatch, t)

            if t % self.print_every == 0:
                #self._visualize()
                self._test(t)


    def run_validation(self):
        self.valid_summary = tf.Summary()
        self._test(0)


    def _train(self, xbatch, ybatch, step):
        _, summ, acc, ls, w1 = self.model.sess.run(
            [self.model.train_step, self.model.summaries, self.model.accuracy, self.model.loss, self.model.W1a],
            feed_dict={self.model.x: xbatch, self.model.y_: ybatch,
                       self.model.learning_rate: self.lr,
                       self.model.keep_prob: self.keep_prob})
        #self.loss_history.append(ls)
        #self.train_acc_history.append(acc)

        # Maybe print training loss
        if self.verbose and step % self.print_every == 0:
            print '(Iteration %d ) loss: %f' % (step + 1, ls)

        self.model.train_writer.add_summary(summ, step)

        if step % self.save_every == 0:
            model_dir = self.model.saver.save(self.model.sess, self.model.logs_path)
            print('Model saved at %s' % model_dir)



    def _test(self, step):
        accuracies = []
        losses = []


        for it in range(self.y_val_data.shape[0] / self.batch_size):
            x_valid_data = self.X_val_data[(it * self.batch_size): (it + 1) * self.batch_size, :, :, :, :]
            yvalid_lbl = self.y_val_data[(it * self.batch_size): (it + 1) * self.batch_size]
            summ, acc, ls = self.model.sess.run([self.model.summaries, self.model.accuracy, self.model.loss],
                                          feed_dict={self.model.x: x_valid_data, self.model.y_: yvalid_lbl, self.model.keep_prob: 1.0})
            accuracies.append(acc)
            losses.append(ls)

        # Take the mean of you measure
        accuracy = np.mean(accuracies)
        loss = np.mean(losses)

        # Create a new Summary object with your measure
        self.valid_summary.value.add(tag="accuracy", simple_value=accuracy)
        self.valid_summary.value.add(tag="loss", simple_value=loss)

        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        self.model.test_writer.add_summary(self.valid_summary, step)
        print('---------------------------')
        print('Test at step %s: \t accuracy:%.4f \t loss:%.4f' % (step, accuracy, loss))
