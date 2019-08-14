#tensorflow import
#numpy는 활용도가 높아서 미리 선언했으나 아직 사용되지는 않음
import tensorflow as tf
import numpy as np

# 학습 모델 예시 : (면의 번호, x좌표, y좌표, 색상) -> [(1,1,1, 'blue')] 

# training data
cube_shape_train = [ 모델1, 모델2, 모델3, 모델4]
# cube_direction_train = '회전방향_돌려지는큐브위치'
cube_direction_train = ["left_top", "left_bottom", "right_top", "right_bottom", "up_left", "up_right", "down_left", "down_right", "rolling_left", "rolling_right"]

# Model parameters
# tf.Variable("학습 모델", tf.float32)

#input과 output을 예측함에 있어 사용될 수 있는 변수들... 임의로 해놓기만함
inputVariable = tf.Variable([0.3], tf.float32)
outputVariable = tf.Variable([-0.3], tf.float32)

# Model input and output
# 입력 큐브 모델
input_cube_shape = tf.placeholder(tf.float32, shape=[None, 3]) #3면을 인식 시킬 것이기 때문에(각각의 면을 object로 하고 총 3개 필요)

# 출력 큐브 회전방향
output_cube_direction = tf.placeholder(tf.float32, shape=[None, 1]) #출력 값은 유일

#출력식을 작성해야함, 이건 예시임
hypothesis = x * W + b

# cost/loss function -> 실질적 학습을 하는 함수
cost = tf.reduce_sum(tf.square(hypothesis - output_cube_direction))  # sum of the squares

#지속적으로 정확한 값을 얻어내기 위해 cost을 minimize 하면서 학습을 진행하는 코드(오차를 줄이는 과정)

# optimizer -> 학습을 진행하는 비율
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

# train -> cost 를 최소화(minimize) 하면서 학습을 시키는 코드(optimizer 활성화 코드)
train = optimizer.minimize(cost)

# training
with tf.Session() as sess:
    #반드시 tensor가 사용하는 변수들을 초기화 시켜줘야한다.
    sess.run(tf.global_variables_initializer())

    #하나의 트레이닝을 하는데 2000번으로 나눠서 진행
    for step in range(2001):
        # 샘플 데이터로 트레이닝 시작하여라
        # 변수를 남기는 것은 내가 보고싶을 경우에 남기는 것임. 필요없으면 변수 없애면됨.
        _, exVar1, exVar2 = sess.run(
            train, inputVariable, outputVariable,
            feed_dict = {input_cube_shape: cube_shape_train, output_cube_direction: cube_direction_train}
            )

        if step % 20 == 0:
            print(step, exVar1, exVar2)

    # evaluate training accuracy
    # 예측하고 싶은 큐드 모델을 넣어서 방향값을 예측
    # run 메소드 속 train은 학습시킬때만 이용하면 됨.
    cube_solution = sess.run(hypothesis, feed_dict={input_cube_shape: 입력 시킬 큐브 모델})
    print("큐브를 돌리실 방향은: {cube_solution} 입니다.")
