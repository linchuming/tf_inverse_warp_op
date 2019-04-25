# tf_inverse_warp_op
The implementation of bilinear inverse warp operation at TensorFlow.

The custom operation supports CPU and GPU devices.

### Test Environment
- Linux (Ubuntu 16.04)
- TensorFlow 1.9
- Cuda 9.0

### Build and Run
- Command`make` and generate `inverse_warp.so`

- Run example:

  ```python
  import tensorflow as tf
  from inverse_warp_op import inverse_warp

  inp = tf.ones([1, 300, 600, 3], tf.float32)
  flow = tf.ones([1, 300, 600, 2], tf.float32)
  warp_res = inverse_warp(inp, flow)
  ```

  ​


### API Doc

```python
inverse_warp(
	x,
  	flow
)
```

**Args:**

- `x`: A `Tensor`. Must be `float32` types. The shape can be 3-D `[height, width, channels]`  or 4-D `[batch, height, width, channles]`.
- `flow`: A `Tensor`. Must be `float32` types. The shape can be 3-D `[height, width, 2]`  or 4-D `[batch, height, width, 2]`.

**Returns:**

A float32 `Tensor`  with the same shape as input `x`.





