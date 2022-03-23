import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

# 학습데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# Keras의 Sequential모델을 선언
model = keras.Sequential([
    # 첫 번째 Layer: 데이터를 신경망에 집어넣기
    keras.layers.Dense(32, activation='relu', input_shape = (1, )),
    # 두번째 층 
    keras.layers.Dense(32, activation='relu'),
    # 세번째 출력층: 예측 값 출력하기
    keras.layers.Dense(1)
])

# 모델을 학습시킬 최적화 방법, loss 계산 방법, 평가 방법 설정
model.compile(optimizer='adam',loss='mse',metrics=['mse', 'binary_crossentropy'])
# 모델 학습
history = model.fit(x,y, epochs = 500, batch_size = 100)

print(model.predict([3]))
print(model.predict([5]))

# 결과를 그래프로 시각화
plt.scatter(x, y, label='y_true')
plt.scatter(x, model.predict(x), label='y_pred')
plt.legend()
plt.show()