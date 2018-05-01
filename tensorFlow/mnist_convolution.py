# mnist多层卷积网络Demo

# import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置量初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# mnist数据集被分成两部分：60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
# 这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，
# 从而更加容易把设计的模型推广到其他数据集上（即：泛化）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # 图像输入向量
y_ = tf.placeholder("float", [None, 10])  # 实际分布

# 第一层卷积由一个卷积接一个max pooling完成。
# 卷积在每个5x5的patch中算出32个特征。卷积的权重张量是[5, 5, 1, 32]，
# 前两个维度是patch的大小（5x5），接着是输入的通道数目（1），最后是输出的通道数目（32）。
W_conv1 = weight_variable([5, 5, 1, 32])

# 对于每一个输出通道都有一个对应的偏置量。故为32。
b_conv1 = bias_variable([32])

# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
# 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积，把几个类似的层堆叠起来，构建一个更深的网络。
# 第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
# 图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout为了减少过拟合而加入。
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 添加一个softmax层，与前面的单层softmax regression一样。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

######################################训练和评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 计算平均数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 在Session里面启动模型
sess = tf.Session()
# 初始化我们创建的变量
sess.run(tf.global_variables_initializer())

# 用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。
# 然后每100次迭代输出一次日志。
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 输出最终的准确率
print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  