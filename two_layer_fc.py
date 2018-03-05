from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np

#Read and prepare training data matrix
train_data = pd.read_csv("train_data_formatted.csv", low_memory=False)
train_data = train_data.as_matrix()
train_labels = pd.read_csv("train_data_labels.csv", low_memory=False)
train_labels = train_labels.as_matrix()

# Read and prepare test data matrix
test_data = pd.read_csv("test_data_formatted.csv", low_memory=False)
test_data = test_data.as_matrix()
test_labels = pd.read_csv("test_data_labels.csv", low_memory=False)
test_labels = test_labels.as_matrix()

print(np.shape(train_data))
print(np.shape(train_labels))
print(np.shape(test_data))
print(np.shape(test_labels))

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 308 
n_classes = 1

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model - Add more layers (ReLU, dropout, fully connected)here
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # fully connected
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])  # fully connected
    # Output fully connected layer
    out_probs = tf.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out']) #output layer
    return out_probs

# Construct model
output_probs = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(-Y*tf.log(output_probs+1e-07) 
                         - (1.0-Y)*tf.log(1.0-output_probs+1e-07))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_data.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = train_data[i*batch_size:min(train_data.shape[0],(i+1)*batch_size), :]
            batch_y = train_labels[i*batch_size:min(train_data.shape[0],(i+1)*batch_size), :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    pred = tf.cast(tf.greater_equal(output_probs, 0.5), "float")
    correct_prediction = tf.equal(pred, Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test_data, Y: test_labels}))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(Y, 0.0), tf.equal(pred,1.0)), "float"))
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(Y, 1.0), tf.equal(pred,1.0)), "float"))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(Y, 1.0), tf.equal(pred,0.0)), "float"))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(Y, 0.0), tf.equal(pred,0.0)), "float"))
    fpv = fp.eval({X: test_data, Y: test_labels})
    tpv = tp.eval({X: test_data, Y: test_labels})
    fnv = fn.eval({X: test_data, Y: test_labels})
    tnv = tn.eval({X: test_data, Y: test_labels})
    print("False Positives:", fpv)
    print("True Positives:", tpv)
    print("False Negatives:", fnv)
    print("True Negatives:", tnv)
    print("True positive rate:", tpv/(tpv+fnv))
    print("False positive rate:", fpv/(fpv+tnv))
