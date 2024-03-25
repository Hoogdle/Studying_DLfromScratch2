### RNN의 문제점

# RNN에서는 BPTT에서 기울기 소실 혹은 기울기 폭발이 일어나기 때문에 장기 의존 관계를 학습하기 어렵다.

### RNN 복습
# h.t = tanh(h.t-1 x W.h + x.t x W.x + b)
# *** W.x와 W.h는 모든 계층에서 동일하다!

### 기울기 소실 또는 기울기 폭발
# 언어 모델은 주어진 단어들을 기초로 다음에 출현할 단어를 예측하는 일을 한다.
# ex)
# Tom was watching TV hin his room. Mary came into the room. Mary said hi to [ ? ]
# 빈칸을 맞추려면 이전 시계열 데이터에 있던 정보들이 RNN 계층의 은닉 상태에 인코딩 하여 보관해둬야 한다.
# 학습에 관점에서 볼 때, 만약 빈칸의 계층의 label이 Tom 이였다면 빈칸 계층의 output값과 label 값의 차이가 Softmax with Loss를 통해 계산된다.
# 이후 해당 손실에 대한 가중치를 갱신하기 위해 역전파가 진행되는데 계층이 길어질수록 기울기가 중간에 사그라들어 가중치 매개변수의 갱신이 전혀되지 않게 된다. 따라서 학습이 진행되지 않게 된다.


### 기울기 소실과 기울기 폭발의 원인
# 역전파에서의 tanh과 matmul의 연산이 기울기 소실과 폭발의 주 원인이 된다.
# tanh을 미분하게 되면 최대값은 x=0 일 때 1이고 0에서 멀어질수록 미분 값이 점점 작아지게 된다.
# 따라서 tanh가 역전파에서 반복됨에 따라 0과 1사이의 값이 계속 곱해져 기울기의 소실이 발생하게 된다.

# 상류로부터 dh라는 기울기가 흘러온다고 했을 때 matmul의 역전파는 dh * Wh.T(transpose)가 된다.
# Wh가 스칼라인 경우 : Wh.T가 반복해서 곱해지기 때문에 Wh가 1보다 크면 지수적으로 증가, 1보다 작으면 지수적으로 감소하게 된다.
# Wh가 행렬인 경우 : 행렬의 특잇값이 1보다 크면 지수적으로 증가, 1보다 작으면 지수적으로 감소할 가능성이 높다
# cf) 행렬의 특잇값 : 데이터가 얼마나 퍼져 있는 정도


### 기울기 폭발 대책

# 기울기 클리핑(gardients clipping)
# if ||g.hat|| >= threshold:
# g.hat = (threshold * g.hat) / ||g.hat||
# g.hat 은 모든 매개변수의 기울기를 하나로 모은 것
# ex) 가중치 W1과 W2의 기울기가 dW1, dW2라 했을 때 g.hat은 dW1과 dW2를 결합한 것이다.

### 기울기 클리핑 구현

import numpy as np

dW1 = np.random.rand(3,3) * 10
dW2 = np.random.rand(3,3) * 10
grads = [dW1,dW2]
max_norm = 5.0

def clip_grads(grads,max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)

    if rate<1:
        for grad in grads:
            grad *= rate

