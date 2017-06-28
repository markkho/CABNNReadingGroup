from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import reader

supervisor = tf.train.Supervisor(logdir="./tmp/mylogs")
raw_data = reader.ptb_raw_data("./data/")
train_data, valid_data, test_data, _ = raw_data

#config
init_scale = 0.1
learning_rate = 1.0
max_grad_norm = 5
num_layers = 2
num_steps = 20
hidden_size = 200
max_epoch = 4
max_max_epoch = 13
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
vocab_size = 10000
train_epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
valid_epoch_size = ((len(valid_data) // batch_size) - 1) // num_steps
test_epoch_size = ((len(test_data) // batch_size) - 1) // num_steps


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(
        hidden_size,
        forget_bias=0.0,
        state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse)

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.name_scope("Train"):
        training_x, training_y = reader.ptb_producer(data=train_data,
                                                     batch_size=batch_size,
                                                     num_steps=num_steps,
                                                     name="TrainInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell() for _ in range(num_layers)],
                state_is_tuple=True
            )

        initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, training_x)

        outputs = []
        state = initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, vocab_size], dtype=tf.float32
        )
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=tf.float32
        )
        logits = tf.matmul(output, softmax_w) + softmax_b

        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

        # use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            training_y,
            tf.ones([batch_size, num_steps], tf.float32),
            average_across_timesteps=False,
            average_across_batch=True
        )

        cost = tf.reduce_sum(loss)
        final_state = state

        lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

        new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        lr_update = tf.assign(lr, new_lr)

