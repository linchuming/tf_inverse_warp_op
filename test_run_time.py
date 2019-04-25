import tensorflow as tf
from inverse_warp_op import inverse_warp
import time
import numpy as np
shape1 = [32, 300, 600, 3]
shape2 = [32, 300, 600, 2]
inp = tf.constant(np.random.rand(*shape1), tf.float32)
flow = tf.constant(np.random.rand(*shape2), tf.float32)
warp_res = inverse_warp(inp, flow)
with tf.device('/cpu:0'):
	warp_res_cpu = inverse_warp(inp, flow)
with tf.Session() as sess:
	sess.run(warp_res)
	sess.run(warp_res)
	start_time = time.time()
	out1 = sess.run(warp_res)
	print(time.time() - start_time)
	out2 = sess.run(warp_res_cpu)
	print(np.max(np.abs(out1 - out2)), np.mean(np.abs(out1 - out2)))