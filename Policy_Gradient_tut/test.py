import tensorflow as tf

sess=tf.Session()

a=tf.one_hot([1,2,3],10)

print(sess.run(a))