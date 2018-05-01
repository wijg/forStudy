from __future__ import absolute_import #绝对引入
from __future__ import division        #精确乘法
from __future__ import print_function

import argparse # Python内置的一个用于命令项选项与参数解析的模块
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

DATA_PATH = None

def main(args):
    #导入数据
    mnist = input_data.read_data_sets(DATA_PATH.data_dir)

    x = tf.placeholder(tf.float32, [None, 784]) #这里的None表示此张量的第一个维度可以是任何长度的
    y_ = tf.placeholder(tf.int64, [None]) #定义损失和优化器

    #为权重值和偏置量赋初值为0
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #计算交叉熵，y 是我们预测的概率分布, y_ 是实际的分布
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    #用梯度下降算法训练你的模型，微调你的变量，不断减少成本
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #训练
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #使用测试集查看训练结果
    # 检测我们的预测是否真实标签匹配，返回一组布尔值
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    # 取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  DATA_PATH, unparsed = parser.parse_known_args()
  # print(DATA_PATH)
  # print(unparsed)
  # print([sys.argv[0]] + unparsed)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
