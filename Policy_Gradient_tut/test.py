import tensorflow as tf

sess=tf.Session()
b=tf.constant([[1,2,3,4],
               [5,6,7,8],
               [9,10,11,12]],dtype=tf.float32)
a=tf.one_hot([1,2,3],4)

c=a*b
d=tf.reduce_sum(c,axis=1)

print(sess.run(a))
print(sess.run(c))
print(sess.run(d))