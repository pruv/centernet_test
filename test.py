import tensorflow as tf

# h = tf.range(0., tf.cast(96, tf.float32), dtype=tf.float32)
# w = tf.range(0., tf.cast(96, tf.float32), dtype=tf.float32)
#
# print(h.shape)
# print(w.shape)
#
# mg_w, mg_h = tf.meshgrid(tf.keras.backend.eval(w), tf.keras.backend.eval(h))
#
# print(tf.keras.backend.eval(h))
# print('----------------')
# print(tf.keras.backend.eval(mg_w))
# print(mg_w.shape)

a = tf.convert_to_tensor([416, 175, 500, 314], dtype=tf.float32)
ngbbox_y = a[..., 0] / 4.0
ngbbox_x = a[..., 1] / 4.0
ngbbox_h = a[..., 2] / 4.0
ngbbox_w = a[..., 3] / 4.0

ngbbox_yx = a[..., 0:2] / 4.0

print(tf.keras.backend.eval(ngbbox_yx))
print(tf.keras.backend.eval(tf.floor(ngbbox_yx)))