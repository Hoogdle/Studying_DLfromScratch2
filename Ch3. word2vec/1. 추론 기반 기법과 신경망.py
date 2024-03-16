### 추론 기반 기법과 신경망 ###
# 추론 기법에서는 신경망이 사용된다.


### 통계 기반 기법의 문제점
# 실제로는 말뭉치의 규모가 상당히 크다.(영어 같은 경우 100만 over)
# 100만 x 100만 차원을 다루는 것은 상당히 비효울적이고 계산 불가능하다...

# 통계 기반 기법 vs 추론 기반 기법
# 통계 기반 기법은 1회만에 단어의 분산 표현을 얻는다
# 추론 기반 기법은 데이터를 미니배치로 나눠 신경망을 학습한다.
# ==> 추론 기반 기법이 훨씬 효율적!

### 추론 기반 기법 개요
# '추론'이 주된 작업, "you [ ] goodbye and i say hello." 에서 주변 단어들을 맥락으로 사용해 빈칸에 들어갈 단어를 추측
# 이러한 추론 문제를 반복해서 풀다보면 단어의 출현 패턴이 학습됨.
# '맥락'(you,goodbye)를 입력하면 모델은 각 단어의 출현 확률을 예측한다. (you=2%,say==90%,goodbye=3%,and=7% ....)
# 말뭉치로 모델이 올바른 추측을 내놓도록 학습을 시키고 그 학습의 결과로 단어의 분산 표현을 얻는 것이 추론 기반 기법의 전체 그림이다.


### 신경망에서의 단어 처리
# 신경망에서는 "you" 나 "say" 같은 단어를 있는 그대로 처리할 수 없다 => 원핫 벡터로!
# "you say goodbye and i say hello."
#   단어          단어 ID              원핫 표현  
#   you             0               (1,0,0,0,0,0,0) 
#   goodbye         2               (0,0,1,0,0,0,0)   

# 총 어휘 수만큼의 원소를 갖는 벡터 준비
# 단어 ID와 동일한 인덱스의 요소만 1로 나머지는 모두 0
# ==> 이를 기준으로 뉴런의 입력층을 만들 수 있다 ==> 신경망으로 처리할 수 있다!

# cf) 신경망은 완전연결계층!

# 완전 연결계층에 의한 변환 구현

import numpy as np

c = np.array([[1,0,0,0,0,0,0]]) #입력 #미니배치 처리를 고려하여 2차원으로
W = np.random.rand(7,3)       #가중치
h = np.matmul(c,W)            #중간 노드
print(h)    #[0.60622165 0.90709527 0.8365695 ]
# 단어 ID가 0인 단어를 원핫벡터로 표현후 입력층으로 사용한 것
# *** c는 원핫벡터이므로 c와W의 행렬곱은 가중치의 행벡터 하나를 '뽑아낸'것! (행벡터 하나를 뽑아내는 것인데 matmul연산은 비효율적이긴함. 뒷장에서 개선)
