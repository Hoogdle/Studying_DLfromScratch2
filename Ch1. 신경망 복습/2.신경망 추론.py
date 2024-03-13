### 신경망의 추론 ###

### 신경망 추론 전체 그림
# 신경망을 간단히 말하면 단순한 '함수' 라고 할 수 있다.
# 완전신경망 : 인접하는 층의 모든 뉴런과 연결 되어 있는 신경망
# 은닉층의 뉴런은 가중치의 합으로 계산된다.

# 미니배치 구현 식
# x : Nx2 (N: 데이터 갯수)
# W : 2x4
# h : Nx4
# h = xW + b

# import numpy as np
# W1 = np.random.randn(2,4) #가중치
# b1 = np.random.randn(4) #편향
# x = np.random.randn(10,2) # 입력
# h = np.matmul(x,W1)+b1 #(b는 10,4로 자동변환, 브로드캐스트)
# 10개의 샘플 데이터 각각을 완전연결계층으로 변환
# x[0] : 0번 째 입력 데이터
# x[1] : 1번 째 입력 데이터
# h[0] : 0번 째 은닉층 뉴런
# h[1] : 1번 째 은닉층 뉴런

# 완전연결계층에 의한 변환은 '선형' 변환. 활성화 함수로 '비선형' 효과를 부여
# 비선형 활성화 함수를 이용함으로써 신경망의 표현력을 높일 수 있다.

# 시그모이드 함수
# 0과 1사이의 실수를 출력
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# a = sigmoid(h) # 은닉층 뉴런에 시그모이드 함수 사용

# 최종 코드
# 해당 교재에서는 '행벡터'를 기준으로
# import numpy as np

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# x = np.random.randn(10,2) # 2차원의 데이터 10개로 미니배치
# W1 = np.random.randn(2,4)
# b1 = np.random.randn(4)
# W2 = np.random.randn(4,3)
# b2 = np.random.randn(3)

# h = np.matmul(x,W1) + b1
# a = sigmoid(h)
# s = np.matmul(a,W2) + b2 # s는 (10,3), 10개의 데이터가 한 번에 처리되었고 3차원의 데이터로 변환됨.
# 3차원이기 때문에 3개의 클래스로 분류할 수 있음. 이 경우 3차원의 출력은 각 클래스에 해당 하는 점수.
# 만약 분류를 한다면 가장 큰 점수(값)을 내뱉는 뉴런에 해당하는 클래스가 예측 결과에 해당



### 계층으로 클래스화 및 순전파 구현
# 이 책에서는
# 완전열결계층에 의한 변환, Affine 계층
# 시그모이드 함수에 의한 변환, Sigmoid 계층
# 기본 변환 수행 메서드, forward()
# 신경망의 다양한 계층을 클래스로 구현, 이렇게 모듈화 하면 레고 블록 조합하듯 신경망 구축 가능

# 이 책의 구현 규칙
# 1. 모든 계층은 forward()와 backward() 메서드를 가진다
# 2. 모든 계층은 인스턴스 변수인 params와 grads를 가진다

# forward()와 backward() 메서드는 각각 순전파와 역전파를 수행
# params는 가중치와 편향 같은 매개변수를 담는 리스트
# gards는 params에 저장된 각 매개변수에 대응하여 해당 매개변수의 기울기를 보관하는 리스트
# 이 구현 규칙을 지키면 확장성이 좋아진다!

### 위의 규칙을 반영하여 코드를 작성한다

### Sigmoid 계층
import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = [] # 시그모이드 계층에는 학습 매개변수가 없으므로 빈 리스트로 초기화

    def forward(self,x): # 주 변환 처리는 forward(x)가 담당
        return 1/(1+np.exp(-x))

### Affine 계층
class Affine:
    def __init__(self,W,b):
        self.params = [W,b] # 초기화시 가중치와 편향을 받음

    def forward(self,x):
        W,b = self.params
        out = np.matmul(x,W)+b
        return out
    
### 입력 x => Affine 계층 => Sigmoid 계층 => Affine 계층 => 출력 s
### TwoLayerNet 클래스

import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        # 모든 가중치를 리스트에 모음
        self.params = []
        for layer in self.layers:
            self.params += layer.params #각 계층의 파라미터들 params에 추가
            # 매개변수들을 하나의 리스트 안에 저장하면 '매개변수 갱신'과 '매개변수 저장'을 손쉽게 처리할 수 있다.

        ### cf
        # a = ['A','B']
        # a += ['C','D']
        # print(a) #['A','B','C','D']

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    
x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)
print(s)
# [[-0.55893961 -1.27580315 -0.00455698]
#  [-0.9076879  -1.31793529 -0.31227234]
#  [-0.8211888  -1.13457836  0.72296682]
#  [-1.13452204 -1.25035245 -0.22606941]
#  [-0.88729348 -1.19392476  0.3087174 ]
#  [-1.02585445 -1.28099728 -0.24433165]
#  [-1.1343992  -1.28309222 -0.3462776 ]
#  [-0.9759271  -1.20630872  0.16885434]
#  [-0.71743795 -1.16199615  0.5829519 ]
#  [-0.56682034 -1.06288412  1.23644557]]