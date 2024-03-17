### word2vec 개선 ###

# 다루는 어휘의 수가 많아지면(입력층의 차원이 커지면) 계산량이 너무 커진다.
# ex) 1000000개의 뉴런을 갖는 입력층, W(in)가 1000000x100 이면 연산량은 100000000이 된다.(too big..)
#     또한 출력층에서도 W(out)이 100x1000000이기 때문에 연산량은 100000000이 된다.(too big....!)
# 즉 문제의 구간은 2곳이다.
# 1. 입력층의 원핫표현 and 가중치(W(in))과의 곱
# 2. 은닉층과 가중치(W(out))과의 곱 and Softmax 계층의 계산

# 1. 입력층의 원핫표현 and 가중치(W(in))과의 곱
# 입력층을 원핫 벡터로 다루기 때문에 다루는 어휘가 많아지면 그만큼 다루는 차원의 수도 커지게 된다.(벡터가 다루는 원소수가 커지게 된다.) => 메모리 손실
# 또한 원핫 벡터로 표시된 벡터를 가중치와 곱 연산을 해야하는데 이것도 상당한 계산 자원이 필요하다. 
# 위 문제들은 Embedding 계층 도입을 해결한다!

# 2. 은닉층과 가중치(W(out))과의 곱 and Softmax 계층의 계산
# 은닉층과 가중치(W(out))과의 연산량은 상당하다.
# 또한 Softmax연산도 다루는 계층이 많아짐에 따라 연산량이 증가하게 된다.
# 위 문제들은 '네거티브 샘플링'이라는 새로운 손실 함수를 도입하여 해결한다!


### Embedding 계층
# CBOW에서 입력 원핫 벡터와 가중치의 곱은 사실상 원핫 벡터가 나타내는 가중치 행렬의 열을 뽑아오는 것이다!
# 사실상 행렬곱을 진행할 필요가 없다!
# Embedding 계층 : 가중치 행렬로부터 단어 ID에 해당하는 추출하는 계층('단어 임베딩' 이라는 용어에서 유래됨)
# 즉 Embedding 계층에 단어 임베딩(분산 표현)을 저장하는 것!



### Embedding 계층 구현
# just 원하는 행 명시!
import numpy as np
W = np.arange(21).reshape(7,3)
print(W)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]
#  [12 13 14]
#  [15 16 17]
#  [18 19 20]]
print(W[2]) #[6 7 8]
print(W[5]) #[15 16 17]

# 여러행을 한 꺼번에 추출하기(미니배치 처리에 활용)
idx = np.array([1,0,3,0])
print(W[idx])
# [[ 3  4  5]
#  [ 0  1  2]
#  [ 9 10 11]
#  [ 0  1  2]]
print(W[1,1]) #4 #indexing 할 때 list와 np.array는 효과가 다르다!

# Embedding 계층의 순전파, 역전파 구현
# 순전파시 W행렬에서 특정 행을 선택, 역전파는 손실계층에서 흘러온 값을 전의 특정 행에 더해주면 된다.
# 더하지 않고 값을 그대로 붙이는 경우 기울기의 소실이 발생하기에 값을 더해줘야 한다!
class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None # 추출할 가중치의 행 인덱스
    
    def forward(self,idx):
        W, = self.params
        self.idx = idx #값일 수도 있고, 벡터(미니배치)일수도 있다.
        out = W[idx]
        return out
    
    def backward(self,dout):
        dW, = self.grads
        dW[...] = 0

        for i,word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        # ==
        # np.add.at(dW,self.idx,dout) #더 빠름!

        return None
# 위의 구현된 Embedding 계층으로 MatMul계층을 대신할 수 있다!(메모리 효율적, 계산 빠름)

