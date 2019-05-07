# 使用Tensorflow构建的麦卡洛克-皮特斯神经元并以Tensorboard进行展示
# Copyright 2018 Denis Rothman MIT License. See LICENSE.

import tensorflow as tf
import numpy as np
import os

PATH = os.getcwd()

LOG_DIR = PATH+ '/output/'



# 1.通过定义线程池来配置优化CPU性能。
#   本便中4个足够了，变量替代了常量
config = tf.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4
)

# 2.定义 x 值，w 权重，b 偏置量，y 权重计算以及 s sigmoid激活函数

x = tf.placeholder(tf.float32, shape=(1, 5), name='x')
w = tf.placeholder(tf.float32, shape=(5, 1), name='w')
b = tf.placeholder(tf.float32, shape=(1), name='b')
y = tf.matmul(x, w) + b
s = tf.nn.sigmoid(y)


# 3.开启一个会话，提供常量作为权重输入
# 感知机（Perceptron），一个可以学习自己权重的神经元，将提供我们现代的自动权重计算
with tf.Session(config=config) as tfs:
    tfs.run(tf.global_variables_initializer())
    
    w_t = [[.1, .7, .75, .60, .20]]
    x_1 = [[10, 2, 1., 6., 2.]]
    b_1 = [1]
    w_1 = np.transpose(w_t)
    
    value = tfs.run(s, 
        feed_dict={
            x: x_1, 
            w: w_1,
            b: b_1
        }
    )
    
print ('value for threshold calculation',value)
print ('Availability of lx',1-value)
         
  
#___________Tensorboard________________________

#with tf.Session() as sess:

Writer = tf.summary.FileWriter(LOG_DIR, tfs.graph)
Writer.close()


def launchTensorBoard():
    import os
    os.system('tensorboard --logdir='+LOG_DIR)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

tfs.close()
# 打开浏览器并访问http://localhost:6006
# 尝试不同的选项。它是一个很有用的工具。
# 在完成后关闭系统窗口
