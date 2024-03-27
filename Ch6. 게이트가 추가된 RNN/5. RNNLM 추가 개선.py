### RNNLM 추가 개선 ###

# RNNLM의 개선 포인트 3가지를 설명


### LSTM 계층 다층화
# 지금 까지는 LSTM을 1층(수직방향)으로만 쌓아서 학습하는 경우만을 봤다.
# LSTM을 쌓아서(2층,3층... 수직방향) 학습을 하면 더 복잡한 패턴을 학습할 수 있게 된다.


### 드롭아웃에 의한 과적합 억제
# LSTM을 깊게 쌓으면 표현력이 풍부한 모델이 만들어지지만 과적합 문제가 발생할 수 있다.
# cf) 과적합
# : 훈련 데이터에만 너무 치중해 학습된 상태, 일반화 능력이 결여된 상태이다.
# 과적합을 억제하는 전통적인 방법이 있다. ex) '훈련 데이터의 양 늘리기', '모델의 복잡도 줄이기', '정규화','드롭아웃'
# 드롭아웃은 훈련시 계층 내의 뉴런 몇개를 무작위로 무시하고 학습하는 일종의 정규화 방법이다.
# LSTM(RNN) 같은 시계열 모델에서는 일반적으로 시계열 방향으로 Dropout을 배치하지 않고 상하 방향, 즉 층 방향으로 Dropout을 배치한다.(시계열 정보가 손실될 수 있기 때문)
# 시계열 방향으로도 Dropout을 배치할 수 있는데 이 때 변형 드롭아웃(같은 계층에 적용되는 드롭아웃 끼리는 공통의 마스크를 이용)으로 드롭아웃을 효과적으로 적용한다.

### 가중치 공유
# Embedding 계층의 가중치와 Affine 계층의 가중치를 연결(공유)하여 매개변수의 수가 크게 줄고 정확도도 크게 향상 시킬 수 있다.

### 개선된 RNNLM 구현

import numpy as np
import sys
sys.path.append('...')
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel

class BeeterRnnlm(BaseModel):
    def __init__(self,vocab_size=10000,wordvec_size=650,hidden_size=650,dropout_ratio=0.5):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx1 = (rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4*H).astype('f')
        lstm_Wx2 = (rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 세 가지 개선
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1,lstm_Wh1,lstm_b1,stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1,lstm_Wh1,lstm_b1,stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T,affine_b)
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2],self.layers[4]]
        self.drop_layers = [self.layers[1],self.layers[3],self.layers[5]]

        self.params, self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self,xs,train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self,xs,ts,train_flg=True):
        score = self.predict(xs,train_flg)
        loss = self.loss_layer(score,ts)
        return loss
    def backward(self,dout=1):
        dout = self.lstm_layers.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

### 매 에폭에서 검증 데이터로 퍼플렉서티를 평가하고 그 값이 나빠졌을 때 학습률을 낮추는 방법으로 좋은 결과를 도출할 수도 있다.
            

