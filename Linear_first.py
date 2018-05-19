import tensorflow as tf

#data set
x_train = [1,2,3]
y_train = [1,2,3]

#H(x)=wx+b
w = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
hypothesis = x_train * w + b

#cost/loss function : H(x)와 data set의 차이 제곱을 최소화
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#아직 요 클래스는 잘 모르겠다
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch
sess = tf.Session()
#Run / Initialize
sess.run(tf.global_variables_initializer())

for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(w), sess.run(b))