### 학습 데이터 준비 ###
# word2vec에 쓰일 학습 데이터 준비

### 맥락과 타깃
# word2vec의 입력은 '맥락'이고 정답 레이블은 '타깃'(맥락안에 들어갈 중앙의 단어)이다.
# 목적은 맥락을 넣었을 때 타깃이 출현할 확률을 높이는 것이다.

# 맥락과 타깃을 만드는 예
#                                           맥락            타깃           
# you say goodbye and I say hello.      you,goodbye         say
# you say goodbye and I say hello.      say,and             goodbye
# you say goodbye and I say hello.      goodbye,I           and
# you say goodbye and I say hello.      and,say             I
# you say goodbye and I say hello.      I,hello             say
# you say goodbye and I say hello.      say,.               hello

# 위 작업을 말뭉치에서 양쪽 끝 단어를 제외하고 모든 단어에 적용
# 맥락의 수는 여러개가 될 수 있으나 타깃은 단 하나!

### 맥락과 타깃을 만드는 코드 구현

import numpy as np
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
print(corpus) #[0 1 2 3 4 1 5 6] 
# cf) corpus는 말뭉치의 word들을 id로 변환한 리스트(중복허용)
# word_to_list는 말뭉치의 word들을 중복을 허용하지 않고 각 단어마다 각각의 id를 부여한 dict
print(id_to_word) #{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

def creat_contexts_target(corpus,window_size=1):
    target = corpus[window_size:-window_size] # window의 인덱스가 곧 타겟의 시작점! # 타깃(정답 레이블) 리스트 추가 #윈도우 사이즈를 이용해 센스있게 추가한 코드
    contexts = [] # 문맥 저장 리스트

    for idx in range(window_size,len(corpus)-window_size): # idx == target의 index
        cs = []
        for t in range(-window_size,window_size+1): # t는 window의 index
            if t == 0:
                continue
            cs.append(corpus[idx+t]) # window는 -window_size~window_size까지 이동하며 target(idx)와 t(window index)를 더한 값을 cs리스트 추가(맥락 추가)
        contexts.append(cs)

    return np.array(contexts), np.array(target)

context,target = creat_contexts_target(corpus,window_size=1)
print(context) # 모델의 입력 데이터들
# [[0 2]
#  [1 3]
#  [2 4]
#  [3 1]
#  [4 5]
#  [1 6]]
print(target) # [1 2 3 4 1 5] #모델의 정답 레이블들

