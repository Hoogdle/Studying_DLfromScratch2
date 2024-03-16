### 단순한 word2vec ###
# word2vec중 CBOW(continuous bag-of-words, CBOW)을 이번 절에서 사용


### CBOW 모델의 추론 처리
# 맥락으로부터 타깃을 추측하는 용도의 신경망(타깃==중앙단어, 맥락==주변단어)
# CBOW 모델로 단어의 분산 표현을 얻어내는 것이 목적
# CBOW 모델의 입력은 맥락, you나 goodbye같은 단어들의 목록, 모델이 처리할 수 있게 입력을 원핫으로 표현
# (해당 교재에서는 window 사이즈를 1로 했기에 2개의 단어인거지 맥락으로 고려할 단어를 늘리면 그 만큼 입력도 늘어남)
# 입력층은 모두 동일한 가중치 W(in)을 거쳐 처리됨.
# 은닉층에서 출력층으로는 가중치 W(out)을 거쳐 처리됨.
# 은닉층은 (선형변환된 입력층들의 합) / (은닉층 노드의 갯수) 이다. (CBOW 특징)
# h1이 은닉층 노드1의 값, h2를 은닉층 노드2의 값 이라 한다면, 은닉층 뉴런은 1/2(h1+h2) (CBOW특징)이다.
# 출력층의 뉴런은 총 7개인데 모두 각 단어에 대응한다. 출력층의 노드는 각 단어의 '점수'를 뜻하며 이는 소프트맥스 함수로 확률로 변환할 수도 있다.
# W(in)이 단어의 분산 표현의 정체이다.(단어의 의미 또한 잘 녹아들어 있다!)

# cf) 인코딩, 디코딩
# 인코딩 : word2vec에서의 은닉층 정보, 인간은 이해할 수 없는 코드(컴퓨터는 이해)
# 디코딩 : 은닉층의 정보로부터 원하는 결과를 얻는 작업, 인간이 이해할 수 있는 정보

### CBOW 모델의 추론과정 구현

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([1,0,0,0,0,0,0])
c1 = np.array([0,0,1,0,0,0,0])

# 가중치 초기화
W_in = np.random.rand(7,3)
W_out = np.random.rand(3,7)

# 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0+h1)
s = out_layer.forward(h)

print(s) 
#[0.85879965 0.31831209 0.83337844 0.8919797  0.8085247  0.34138898  0.89474501]

