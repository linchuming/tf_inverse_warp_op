import tensorflow as tf
import inverse_warp_op
import numpy as np

shape1 = [1, 10, 10, 3]
shape2 = [1, 10, 10, 2]
d1 = np.random.randn(*shape1)
d2 = np.random.randn(*shape2)
inp = tf.constant(d1, tf.float32)
flow = tf.constant(d2, tf.float32)
with tf.device('/gpu:0'):
	warp_res = inverse_warp_op.inverse_warp(inp, flow)
with tf.Session() as sess:
	inp_grad_err = tf.test.compute_gradient_error(
		inp, shape1, warp_res, shape1)
	flow_grad_err = tf.test.compute_gradient_error(
		flow, shape2, warp_res, shape1)

print('inp_grad_error:', inp_grad_err)
print('flow_grad_error:', flow_grad_err)

with tf.device('/cpu:0'):
	warp_res_cpu = inverse_warp_op.inverse_warp(inp, flow)
	grad_cpu = tf.gradients(warp_res_cpu, [inp, flow])
grad_gpu = tf.gradients(warp_res, [inp, flow])
with tf.Session() as sess:
	r1, r2, g1, g2 = sess.run([warp_res, warp_res_cpu, grad_gpu, grad_cpu])
	print(np.mean(np.abs(r1 - r2)))
	print(np.mean(np.abs(g1[0] - g2[0])), np.mean(np.abs(g1[1] - g2[1])))