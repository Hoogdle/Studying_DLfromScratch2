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


### SVD 차원감소 구현
# SVD는 넘파이의 linalg 모듈을 사용해 구현 가능(linalg는 선형대수의 약어)

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

# SVD
U,S,V = np.linalg.svd(W)

print(C[0]) #[0 1 0 0 0 0 0]
print(W[0]) #[0.        1.8073549 0.        0.        0.        0.        0.       ]
print(U[0]) #SVD 자체, 밀집벡터로 변환됨. #[-1.1102230e-16  3.4094876e-01 -1.2051624e-01 -3.8857806e-16  0.0000000e+00 -9.3232495e-01  8.7683712e-17]

# 벡터의 차원을 감소시키려면 just 원하는 만큼만 뽑으면 됨.
print(U[0][:2]) #[-1.1102230e-16  3.4094876e-01] # 2개만 뽑음

# 그래프로 나타내기
for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id,0],U[word_id,1])) #x,y 지점(벡터, U[word_id,0]과 U[word_id,1]이 나타내는 벡터)에 word에 담긴 텍스트를 그리도록 함

plt.scatter(U[:,0],U[:,1],alpha=0.5) # U 행렬 중 (전체열 and 0행)과 (전채행 and 1행)를 그래프에 표시함. 투명도는 0.5
plt.show() # 그래프 나타남

# 지금은 데이터셋이 작아서 합당하지 않은 결과가 나오지만 앞으로는 PTB라는 큰 말뭉치를 사용하여 실험 수행


### PTB(Penn Treebank) 데이터셋
# PTB는 말뭉치중 하나
# 희소한 단어는 <unk>로 치환 (unknown)
# 구체적인 숫자는 N으로 치환
# <eos> 문장의 끝


# PTB 사용
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train') #train,valid,test의 값을 줄 수 있음

print('말뭉치의 크기 :',len(corpus))
print(f'corpus[:30] : {corpus[:30]}')
print()
print(f'id_to_word[0] : {id_to_word[0]}')
print(f'id_to_word[1] : {id_to_word[1]}')
print(f'id_to_word[2] : {id_to_word[2]}')
print()
print(f'word_to_id["car"] : {word_to_id["car"]}')
print(f'word_to_id["happy"] : {word_to_id["happy"]}')
print(f'word_to_id["lexus"] : {word_to_id["lexus"]}')

# 말뭉치의 크기 : 929589
# corpus[:30] : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29]

# id_to_word[0] : aer
# id_to_word[1] : banknote
# id_to_word[2] : berlitz

# word_to_id["car"] : 3856
# word_to_id["happy"] : 4428
# word_to_id["lexus"] : 7426



# ptb 데이터로 벡터만들고 평가하기

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

C = create_co_matrix(corpus,vocab_size,window_size)
W = ppmi(C, verbose=True)

try:
    from sklearn.utils.extmath import randomized_svd #for 빠른 연산
    U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter=5,random_state=None)
except ImportError:
    U,S,V = np.linalg.svd(W) #위에꺼 안 되면 어쩔 수 없이 느린 연산

word_vecs = U[:,:wordvec_size]

querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query,word_to_id,id_to_word,word_vecs,top=5)


# [query] you
#  i: 0.7016294002532959
#  we: 0.6388040781021118
#  anybody: 0.5868049263954163
#  do: 0.5612815022468567
#  'll: 0.512611985206604

# [query] year
#  month: 0.6957005262374878
#  quarter: 0.691483736038208
#  earlier: 0.6661214232444763
#  last: 0.6327787041664124
#  third: 0.6230477094650269

# [query] car
#  luxury: 0.6767407655715942
#  auto: 0.6339930295944214
#  vehicle: 0.597271203994751
#  cars: 0.5888376235961914
#  truck: 0.5693157911300659

# [query] toyota
#  motor: 0.7481387853622437
#  nissan: 0.7147319912910461
#  motors: 0.6946365833282471
#  lexus: 0.6553674936294556
#  honda: 0.6343469619750977
    

### 정리
# 컴퓨터에게 '단어의 의미'를 이해시키는 것을 메인 포인트로 이번 챕터가 진행됨.
# 시소러스는 많은 인력,표현력의 한계... => 통계 기반 기법
# 통계 기반 기법은 말뭉치로부터 단어의 의미를 자동으로 추출하고 그 의미를 벡터로 표현
# (window와 단어 count를 기준으로) 동시발생 행렬을 만듦 => PPMI 행렬로 변환 => SVD로 차원을 감소, 단어의 분산 표현(비슷한 단어들은 벡터 공간에서도 가까이 모여 있음) => cos 유사도로 유사성 확인