### 어텐션의 구조 ###

# seq2seq를 한층 더 강력하게 해줌
# 어텐션 덕분에 seq2seq는 필요한 정보에만 '주목'할 수 있게 된다.

### seq2seq의 문제점
# Encoder가 시계열 데이터를 인코딩, Encoder의 출력은 '고정 길이 벡터'이다.
# 입력 문장이 아무리 길어도 길이가 고정되어 있기 때문에 한계가 존재한다.(문장의 길이가 길어지면 필요한 정보가 벡터에 모두 담기지 못한다)

### Encoder 개선
# Encoder의 출력 길이는 입력 문장의 길이에 따라 변하는 것이 좋다.
# => Enocder의 LSTM 계층의 모든 은닉 상태 벡터를 모두 이용!
# 각 시각의 은닉 상태 벡터를 모두 이용하면 입력된 단어와 동일한 차원(크기의) 행렬(은닉층의 set)을 얻을 수 있다.
# ex) 단어 5개가 입력 => 은닉층 5개 => Encoder는 5개의 은닉층을 행렬의 형태로 출력 (고정 길이에서 해방)

# Encoder의 각 LSTM 계층의 은닉 상태는 해당 계층의 input 단어의 영향을 가장 많이 받게 된다. 
# 따라서 Encoder가 출력하는 hs 행렬은 각 단어에 해당하는 벡터들의 집합이라고도 볼 수 있다.


### Decoder 개선(1)
# 문장 번역을 예로 들자면 "나는 고양이로소이다" => "I am a cat"으로 번역을 할 때
# 인간은 'I' 단어는 '나는'에 'cat'단어는 '고양이'에 주목하여 단어의 변환을 할 것이다.
# 즉 Decoder의 각 단어는 Encoder의 단어 set중 주목(attention) 해야할 단어를 필요로한다(존재한다).
# 우리의 목표는 필요한 정보에만 주목하여 해당 정보로부터 시계열 변환을 수행하는 것이다. == 어텐션

# 필요한 정보에만 주목하려면 '선택 작업'이 필요하다. 선택을 할 때 선택된 단어만 가지고오는 것은 수학적으로 미분할 수가 없다(역전파 불가 => 학습 불가)
# => 모든 것을 선택하고 가중치 별로 계산하여 합하자!

import numpy as np
T, H = 5,4
hs = np.random.randn(T,H)
a = np.array([0.8,0.1,0.03,0.05,0.02])

ar = a.reshape(5,1).repeat(4,axis=1)
print(ar.shape) #(5, 4)

t = hs*ar
print(t.shape) #(5, 4)

c = np.sum(t,axis=0)
print(c.shape) #(4,)

# attention을 할 때 Decoder에서의 단어를 기준으로 Encoder 단어마다의 확률이 계산되게 된다. 각 확률을 해당단어끼리 곱해주고 곱한 결과를 모두 더해 하나의 벡터로 만든다.
# == 맥락 벡터

N,T,H = 10,5,4
hs = np.random.randn(N,T,H)
a = np.random.randn(N,T)
ar = a.reshape(N,T,1).repeat(H,axis=2)
# ar = a.reshape(N,T,1) # 브로드캐스트를 사용하는 경우

t = hs*ar
print(t.shape)
# (10, 5, 4)

c = np.sum(t,axis=1)
print(c.shape) 
#(10, 4)


# cf)
# sum과 repeat는 정반대의 과정
# sum의 역전파 == repeat의 순전파
# repeat 역전파 == sum의 순전파

class WeightSum:
    def __init__(self):
        self.params, self.grads = [],[]
        self.cache = None

    def forward(self,hs,a):
        N,T,H = hs.shape

        ar = a.reshape(N,T,1).repeat(H,axis=2)
        t = hs*ar

        c = np.sum(t,axis=1)

        self.cache = (hs,ar)
        return c

    def backward(self,dc):
        hs,ar =self.cache
        N,T,H = hs.shape

        # sum의 역전파
        dt = dc.reshape(N,1,H).repeat(T,axis=1)
        dar = dt * hs
        dhs = dt * ar

        # repeat의 역전파
        da = np.sum(dar,axis=2) 

        return dhs,da

### Decoder 개선(2)
# 그렇다면 각 단어의 중요도를 나타내는 a는 어떻게 구할 수 있을까?
# => 데이터로부터 학습할 수 있도록 준비하자!
    
# Decoder의 LSTM 계층의 은닉 상태 벡터를 h라 하면 Encoder의 벡터 set, hs와 내적을 통해 유사도를 측정한다.
# h(단어)의 벡터와 hs[0],hs[1],hs[2],....hs[T]의 내적을 통해 각각의 유사도를 측정하고 이를 softmax를 통해 정규화하여 단어의 중요도를 나타내는 a를 구한다.

import sys
sys.path.append('...')
from common.layers import Softmax
import numpy as np

N,T,H = 10,5,4
hs = np.random.randn(N,T,H)
h = np.random.randn(N,H)
hr = h.reshape(N,1,H).repeat(T,axis=1)
# hr = h.reshape(N,1,H) # 브로드캐스트를 사용하는 경우


########################### 내적 과정 ###########################
t = hs * hr #원소별 곱
print(t.shape) #(10,5,4)

s = np.sum(t,axis=2)
print(s.shape) #(10,5)
########################### 내적 과정 ###########################

softmax = Softmax()
a = softmax.forward(s)
print(a.shape) #(10,5)


### AttentionWeight Class

import numpy as np
import sys 
sys.path.append('...')
from common.np import *
from common.layers import Softmax

class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [],[]
        self.softmax = Softmax()
        self.cache = None
    
    def forward(self,hs,h):
        N,T,H = hs.shape

        hr = h.reshape(N,1,H).repeat(T,axis=1)
        t = hs * hr
        s = np.sum(t,axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs,hr)
        return a
    
    def backward(self,da):
        hs,hr = self.cache
        N,T,H = hs.shape

        ds = self.softmax.backward(da)
        dt = t.reshape(N,T,1).repeat(H,axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr,axis=1)

        return dhs,dh
    

### Decoder 개선(3)
# 위의 개선점들을 하나로 
# 과정을 크게 2가지로 정리하면 다음과 같다.
# (1) Attention Weight
# 1. LSTM의 은닉 상태 h(N,H) 를 hr(N,T,H)로 변환
# 2. hr과 hs를 원소별 곱(내적 과정 중 하나) => t1(N,T,H) 생성
# 3. t1을 2차원(행렬)이 되도록 sum(내적 과정 중 하나) (N,T)
# 4. t1을 score 정보로 보고 softmax 계산 => a(N,T) 확률로 변환

# (2) Weight Sum
# 1. 계산을 위해 a를 확장 =>ar(N,T,H) 
# 2. hs와 ar과 원소별 곱 => t2(N,T,H) 
# 3. t2의 모든 시계열 데이터를 sum하여 하나의 벡터로 만듦 => c(N,H)
    

# Attention Weight 계층은 Encoder가 출력하는 각 단어의 벡터 hs에 주목하여 해당 단어의 가중치 a를 구함
# Weight Sum 계층은 a와 hs의 가중합을 구하고 해당 결과를 맥락 벡터 c로 출력
# ==> Attention 계층
    
# Attention 계층 구현

class Attention:
    def __init__(self):
        self.params,self.gards = [],[]
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self,hs,h):
        a = self.attention_weight_layer.forward(hs,h)
        out = self.weight_sum_layer(hs,a)
        self.attention_weight = a # 각 단어의 가중치를 나중에 참조할 수 있도록
        return out
    
    def backward(self,dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1,dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1 # hs가 Weight Sum 계층과 Attention Weight 계층에 한 번씩 적용된다. 역전파시에는 더해주기!
        return dhs,dh

# Attention의 출력값인 맥락 벡터 c와 LSTM의 출력값 h, 2개의 벡터가 연결(concatenate) 되어 Affine 계층의 입력으로 들어간다.
# Affine의 출력값은 임베딩 벡터이며 임베딩 벡터에 Softmax 연산을 가해 가장 높은 단어를 해당 계층의 출력 단어로 한다.
# Affine 계층의 출력은 배치 사이즈가 1이라고 가정하면 벡터형태이며 vocab의 단어 갯수만큼의 차원을 가진다.
# ex) 배치 사이즈 = 1, vocab의 단어 갯수 = 2056개 => Affine이 적용된 LSTM Decoder Affine 계층의 출력 = 2056개의 entry를 가지는 벡터(2056차원의 벡터)
# 해당 벡터에 softmax 연산을 가해 확률로 변환하고 가장 확률이 높은 entry(단어)를 예측 단어(다음 단어)로 결정한다.

class TimeAttention:
    def __init__(self):
        self.params, self.grads = [],[]
        self.layers = None
        self.attention_weights = None
    
    def forward(self,hs_enc,hs_dec):
        N,T,H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []
        
        for t in range(T):
            layer = Attention()
            out[:,t,:] = layer.forward(hs_enc,hs_dec[:,t,:])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out
    def backward(self,dout):
        N,T,H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:,t,:])
            dhs_enc += dhs
            dhs_dec[:,t,:] = dh

        return dhs_enc, dhs_dec


    



