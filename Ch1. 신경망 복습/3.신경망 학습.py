### 신경망의 학습 ###
# 학습되지 않은 신경망은 '좋은 추론'을 해낼 수 없다. 따라서 학습을 수행한 후 학습된 매개변수로 추론을 수행하는 것이 일반적이다.
# 추론 : 다중 클래스 분류 등의 문제에 답을 구하는 작업
# 학습 : 최적의 매개변수 값을 찾는 작업


### 손실함수
# 학습 단계의 특정 시점에서 신경망의 성능을 나타내는 척도 == '손실'(스칼라)
# 손실함수를 통해 손실 값을 구한다. cf) 다중클래스 분류 ==> 손실함수 == 교차 엔트로피 오차
#  앞선 모델에 소프트맥스 계층과 크로스 엔트로피 계층을 추가해보자 구성은 다음과 같다.
# input x => Affine => Sigmoid => Affine => Softmax => CrossEntropy with Lable => Loss
# Softmax의 출력은 확률로써 해석할 수 있다.(모든 노드의 합은 1)
# 이 책에서는 Softmax 함수와 Cross Entrophy 를 묶어 Softmax with Loss 계층 하나로 구현(두 계층을 통합하면 역전파 계산이 쉬워진다.)
# input x => Affine => Sigmoid => Affine => Softmax with Cross Entropy(with label) => Loss

### 미분과 기울기
# 신경망의 학습 목표는 손실을 최소화하는 매개변수를 찾는 것. => 미분과 기울기 중요!
# L을 스칼라 f를 함수 x를 벡터라 했을 때
# L에 대한 x의 미분을 구하면 x의 모든 원소를 L에대해 미분한 벡터이다.
# mxn 행렬 W에 대해서 L는 스칼라, g는 함수 일때 L를 W에 대헤 미분한 것은 L에대해 W의 모든 원소를 미분한 것과 같다. 
# 즉, 같은 형상의 행렬을 가지며, 행렬과 기울기의 행렬의 형상이 같은 성질을 이용하면 연쇄 법칙을 쉽게 구현 할 수 있다.

### 연쇄 법칙
# 학습 시 신경망에 데이터를 주면 손실을 출력한다.
# 우리가 얻고 싶은 것은 각 매개변수에 대한 손실의 기울기이다. 해당 기울기를 얻는다면 매개변수의 갱신이 가능하다.
# 역전파(Back-Propagation)으로 각 매개변수에 대한 손실의 기울기를 얻을 수 있다.
# 역전파를 이해하는 열쇠는 '연쇄 법칙(Chain Rule)'이다 (연쇄 법칙 == 합성함수에 대한 미분의 법칙)
# y=f(x), z=g(y) 두 함수가 있을 때 z = g(f(x)) 가 되어 최종 출력은 두 함수의 조합 
# 여기서 x에 대한 z의 미분은 ∂z/∂x = ∂z/∂y * ∂y/∂x 이다.
# 즉 우리가 다루는 함수가(계층)이 아무리 많고 복잡해도 그의 미분은 개별 함수의 미분을 이용해 구할 수 있다.


### 계산 그래프
# 덧셈 노드 
# z = ax+by에서 ∂z/∂x = a, ∂z/∂y =b 
# 즉 상류로 부터 받은 값에 변수의 계수를 곱함

# 곱셈 노드
# z = x*y 에서 ∂z/∂x = y, ∂z/∂y =x
# 즉 상류로 부터 받은 값에 스위칭 한 변수의 값을 곱함
 
# 분기 노드
# 한 선에서 두 선으로 갈라지는 노드. 이 때 각 선에 같은 값이 복제되어 분기한다.(==복제노드)
# 역전파는 상류에서 온 기울기들의 합

# Repeat 노드
# 분기노드를 일반화, N개로 분기한 노드
# 순전파시 같은 값이 그대로 N개의 노드로 가며 역전파시 N개의 상류에서 온 값을 모두 더한 값
# import numpy as np
# D,N = 8,7
# x = np.random.randn(1,D)
# y = np.repeat(x,N,axis=0) #np.repeat() 원소 복제 수행, 배열 x를 N번 복제, axis를 지정하여 원하는 축 방향으로 복제 조정 가능
# dy = np.random.randn(N,D)
# dx = np.sum(dy,axis=0,keepdims=True) #역전파에서의 총합 구하기 axis로 어느축 방향으로 합을 구할지 지정,keepdims로 차원수 유지 : True:(1,D) False:(D)

# Sum 노드
# 순전파시 모든 노드의 값을 하나로 역전파시 하나의 노드의 값을 그대로 모든 노드에 분배
# import numpy as np
# D,N =8,7
# x = np.random.randn(N,D) #입력
# y = np.sum(x,axis=0,keepdims=True) #순전파
# dy = np.random.randn(1,D) # 무작위 기울기
# dx = np.repeat(dy,N,axis=0) #역전파
# sum과 repeat은 반대관계

# MatMul 노드
# 증명과정 까다로움 
# 곱셈노드의 Idea를 갖고옴(입력값을 스와핑해서 상류의 값의 곱하기)
# x = NxD, W = DxH, y = NxH 일 때
# ∂L/∂x = ∂L/∂y * Wt (Transpose) ∂L/∂W = xt(transpose) * ∂L/∂y
import numpy as np

class MatMul:
    def __init__(self,W):
        self.params=[W]
        self.grads=[np.zeros_like(W)] #np.zeros_like() 어떤 변수만큼의 사이즈인 0으로 가득 찬 리스트 반환
        self.x = None
    def forward(self,x):
        W, = self.params
        out = np.matmul(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout,W.T)
        dW = np.matmul(self.x.T,dout)
        self.grads[0][...] = dW #grads[0] : 얕은 복사, grads[0][...] : 깊은 복사(배열이 가리키는 메모리위치 고정, 그 위치에 원소들을 덮어씌움)
        # a와 b모두 리스트일 때, a=b : a가 b와 같은 위치를 가리킴, a[...]=b a의 메모리 위치는 유지, a가 가리키는 메모리에 b의 원소가 복제됨.(깊은복사)
        return dx
