### 통계 기반 기법 개선하기 ###

### 상호 정보량
# 동시발생 행렬의 원소는 두 단어가 동시에 발생한 횟수를 나타낸다.
# the car 과 drive car 에서
# car는 drive와 더 연관이 있어 보이지만 the car이라는 '문장의 량'이 더 많기 때문에 car가 drive가 아닌 the와 더 유사하다고 판단한다(문제점)
# => 점별 상호정보량(Pointwise Mutual Information - PMI) 척도로 위 문제 해결!
# PMI 척도는 두 단어가 동시에 발생할 확률도 고려(분자)하되, 각 단어가 나타날 확률도 고려(분모에서)하여 위의 문제를 해결한다.
# 즉 car가 drive보다 빈도수가 높아 the car의 세트가 drive car의 세트보다 높을 수 있더라도 car의 빈도수와 drive의 빈도수가 분모로써 고려됐을 때의 값은 drive car가 더 높을 수 있다.

# PIM(x,y) = log2(P(x,y)/P(x)P(y)) = log2((C(x,y)/N)/((C(x)/N)(C(y)/N))) = log2((C(x,y)*N)/(C(x)C(y)))
# log2 : 밑이 2인 로그를 씌운다는 것 / P(x,y) x와y 단어가 동시에 발생할 확률 / C(x,y) x와y가 동시에 발생한 갯수

# 만약 두 단어가 동시에 발생하는 경우가 존재하지 않으면 log2(0)으로 -∞가 된다.
# => 양의 상호정보량(Positive PMI - PPMI)로 해결!
# PPMI(x,y) = max(0,PMI(x,y)) #PMI(x,y)의 값이 0보다 작으면 0으로 반환, 0보다 크면 그대로 반환

### ppmi 구현

# C 동시발생 행렬, verbose 진행상황 출력 여부, np.log2(0)이 되는 것을 막기위해 eps
# 동시발생 행렬에 대해서만 PPMI 행렬을 구할 수 있도록 단순화한 코드(근삿값을 구하도록)
# => C(x) = i.∑ C(i,x) / C(y) = i.∑ C(i,y) / N = i.j.∑ C(i,j)
# Original
# N = corpus 단어의 수, C(x) = x가 발생한 수 / N, C(y) = y가 발생한 수 / N
import numpy as np

def ppmi(C,verbose=False,eps=1e-8): 
    M = np.zeros_like(C,dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0) # 아랫방향으로 모두 add (각 단어의 빈도수 구하기)
    total = C.shape[0]*C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[i]*S[j]) + eps)
            M[i,j] = max(0,pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print(f'{100*cnt/total : .1f}%%완료')
    return M


### 실전 적용
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3) # 유효 자릿수를 세 자리로 표시
print('동시발생 행렬')
print(C)
print('-'*50)
print('PPMI')
print(W)

# 동시발생 행렬
# [[0 1 0 0 0 0 0]
#  [1 0 1 0 1 1 0]
#  [0 1 0 1 0 0 0]
#  [0 0 1 0 1 0 0]
#  [0 1 0 1 0 0 0]
#  [0 1 0 0 0 0 1]
#  [0 0 0 0 0 1 0]]
# --------------------------------------------------
# PPMI
# [[0.    1.807 0.    0.    0.    0.    0.   ]
#  [1.807 0.    0.807 0.    0.807 0.807 0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.    1.807 0.    1.807 0.    0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.807 0.    0.    0.    0.    2.807]
#  [0.    0.    0.    0.    0.    2.807 0.   ]]

### PPMI의 문제점
# 말뭉치의 어휘 수가 증가함에 따라 각 단어의 벡터의 차원 수도 증가.(말뭉치 어휘수가 10만 => 벡터 10만 차원....what!)
# PPMI의 결과를 봤을 때 대부분 0, 즉 대부분이 중요하지 않는데 사용됨. ==> 차원 감소 필요!



### 차원 감소(Dimensionality Reduction)
# 중요한 정보를 유지하면서 벡터의 차원을 줄이는 방법

# 직관적인 예시
# 1. 2차원의 데이터가 존재할 때 데이터의 분포(뻗은 모양)를 살펴 가장 적합한 축을 하나 생성한다
# 2. 해당 축으로 모든 데이터를 정사영 한다. 각 데이터의 값은 새로운 축으로 정사영된 값으로 변경된다.
# 적합한 축을 찾아내는 것이 가장 중요하다!

# cf) 희소행렬, 희소벡터
# : 대부분이 0인 행렬, 벡터
# 차원 감소의 핵심은 희소벡터에서 중요한 축을 찾아내어 더 적은 차원으로 다시 표현하는 것
# 차원 감소(정사영)의 결과로 원소 대부분이 0이 아닌 값으로 구성된 '밀집벡터'로 변환
# 이 조밀한 벡터야말로 우리가 원하는 단어의 분산 표현

# 특잇값분해(Singular Value Decompositon - SVD)
# 차원을 감소시키는 방법 중 하나
# SVD 는 임의의 행렬을 세 행렬의 곱으로 분해한다.
# X = USV.T # V transpose # U와 V는 직교행렬(열 벡터는 서로 직교) # S는 대각행렬로 대각 성분 외에는 모두 0인 행렬
# X 는 단어벡터의 matrix (각 행은 각 단어의 벡터들)
# U는 직교행렬 이므로 어떤 공간의 축(기저)를 형성한다. => U를 '단어 공간'으로 취급할 수 있음
# S는 대각행렬로 대각성분에는 '특잇값'이 큰 순서대로 나열되어 있다. 특잇값은 '해당 축'의 중요도라고 간주할 수 있다.

# 차원 감소의 원리는 다음과 같다 (키 포인트는 S!)
# S는 대각성분이 큰 순으로 나열되어 있다. 이 중 중요하지 않은(뒷부분) 성분을 버린다 => S'
# S'로 변하게 되면 행렬의 곱을 맞추기 위해 U와 Vt가 변할 수 밖에 없다 => U' Vt'
# 이렇게 되면 U의 열벡터(단어의 기저 벡터)가 소실되게 되는것! == 차원감소



