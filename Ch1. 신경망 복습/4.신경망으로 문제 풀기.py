### 신경망 구현
# 은닉층이 하나인 신경망 구현

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.layers import Affine,Sigmoid,SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size #입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수

        #가중치와 편향 초기화
        W1 = 0.01 * np.random.rand(I,H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H,O)
        b2 = np.zeros(O)

        # 계층 생성
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params,self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self,x,t):
            score = self.predict(x)
            loss = self.loss_layer(score,t)
            return loss
    def backward(self,dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout



### 학습용 코드

import sys
sys.path.append('...')
import numpy as np
from common.optimizer import SGD # 책에서 제공하는 라이브러리
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet # 앞선 모델

# 하이퍼 파라미터
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 읽기, 모델과 옵티마이저 생성
x,t = spiral.load_data()
model = TwoLayerNet(input_size=2,hidden_size=hidden_size, output_size = 3)
optimizer = SGD(lr=learning_rate)

# 학습에 사용하는 변수
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0 
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 데이터 뒤섞기
    idx = np.random.permutation(data_size) #np.random.permutation(10) => array([7,6,8,3,5,0,4,1,9,2])
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 기울기를 구해 매개변수 갱신
        loss = model.forward(batch_x,batch_t)
        model.backward()
        optimizer.update(model.params,model.grads)

        total_loss += loss
        loss_count += 1

        # 정기적으로 경과 출력
        if(iter+1)%10 ==0:
            avg_loss = total_loss / loss_count
            print('|에폭 %d| 반복 %d/%d| 손실 %.2f|'%(epoch+1,iter+1,max_iters,avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0,0


### 해당 교재에서는 위의 코드를 하나의 클래스(Train)으로 사용!(자주 사용됨)
# Trainer는 (모델,옵티마이저)를 인수로 받음
# fit()으로 학습시작 
# fit의 파라미터
# x : 입력데이터
# t : 정답레이블
# max_poch(=10) : 학습을 수행하는 에폭 수
# batch_size(=32) : 미니배치 크기
# eval_interval(=20) : 결과를 출력하는 간격
# max_grad(=None) : 기울기 최대 노름

### 사용예시
import sys
sys.path.append('...')
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0
x,t = spiral.load_data()
model = TwoLayerNet(input_size=2,hidden_size=hidden_size,output_size=3)
optimizer = SGD(learning_rate)

trainer = Trainer(model,optimizer)
trainer.fit(x,t,max_epoch,batch_size,eval_interval=10)
trainer.plot() # fit()에서 기록한 손실을 그래프로 보여줌

