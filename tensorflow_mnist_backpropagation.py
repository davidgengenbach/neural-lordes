from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

IMAGE_PIXELS = 784

HIDDEN_SIZE = 300

inputVector = tf.placeholder(tf.float32, [None, 784])
W_1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, HIDDEN_SIZE],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))))
b1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))
hiddenLayer = tf.nn.relu(tf.matmul(inputVector, W_1)+b1) #activation function
W_2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, 10],
                            stddev=1.0 / math.sqrt(float(HIDDEN_SIZE))))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(hiddenLayer, W_2) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#cross_entropy_hidden = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(hiddenLayer), reduction_indices=[1]))
init = tf.initialize_all_variables()
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={inputVector: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={inputVector: mnist.test.images, y_: mnist.test.labels}))
