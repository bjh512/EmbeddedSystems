import tensorflow as tf

#placeholder
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#H(x)=wx+b
w = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
hypothesis = X * w + b

#cost/loss function : H(x)와 data set의 차이 제곱을 최소화
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#아직 요 클래스는 잘 모르겠다
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch
sess = tf.Session()
#Run / Initialize
sess.run(tf.global_variables_initializer())

for step in range(2001):
	#여러가지 한 번에 run 가능
	cost_result, w_result, b_resut, _ = sess.run([cost, w, b, train], feed_dict={X : [1, 2, 3], Y : [1, 2, 3]})
	if step % 20 == 0:
		print(step, cost_result, w_result, b_resut)