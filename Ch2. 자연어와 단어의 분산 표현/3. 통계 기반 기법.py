### 통계 기반 기법 ### 

# 말뭉치(corpus) : 대량의 텍스트 데이터
# 말뭉치 안의 데이터들은 사람이 쓴 글이기에 자연어에 대한 사람의 '지식'이 충분히 담겨있다고 볼 수 있다.
# ex) 문장을 쓰는 방법, 단어를 선택하는 방법, 단어의 의미
# 통계 기반 기법은 말뭉치에서 자동으로, 효율적으로 핵심을 추출하는 것이다.


### 파이썬을 말뭋이 전처리하기
# 전처리 : 텍스트 데이터를 단어로 분할하고 분할된 단어들을 단어 ID 목록으로 변환하는 것

text = 'You say goodbye and i say hello.' # 말뭉치 # 실전에는 이러한 말뭉치가 수천~수만개
text = text.lower()
text = text.replace('.',' .')
print(text) #you say goodbye and i say hello .
words = text.split()
print(words) #['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']

# 단어 단위로 분할되어 다루기 쉬워졌지만, 텍스트 그대로 조작하기에는 불편 => 단어마다 ID 부여
# ID의 리스트로 이용할 수 있도록 손질!

word_to_id = {} #단어에서 ID변환 담당
id_to_word = {} #ID에서 단어변환 담당

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(id_to_word) #{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
print(word_to_id) #{'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}

# id를 활용해 단어를 검색하거나, 단어를 통해 id를 검색!
print(id_to_word[1]) #say
print(word_to_id['say']) #1

# 단어목록 => ID 목록
import numpy as np
corpus = [word_to_id[w] for w in words] 
print(corpus) #[0, 1, 2, 3, 4, 1, 5, 6] #리스트
corpus = np.array(corpus)
print(corpus) #[0 1 2 3 4 1 5 6] 넘파이 배열

# 위의 프로세스를 하나의 함수로(preprocess)
def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split() # split은 리스트형을 리턴
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])

    return corpus,word_to_id,id_to_word

# 사용예시
text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
# corpus == ID 목록
# word_to_id == 단어 : ID
# id_to_word == ID : 단어
# 전처리 과정 완료!(말뭉치를 다룰 준비 완료!) => 말뭉치를 사용해 단어의 의미를 추출해야함!



### 단어의 분산 표현
# 모든 색은 각각의 이름이 있다(코발트 블루, 싱크레드) 
# 이름대로 색을 표현하는 방법, RGB(벡터)를 사용하여 색을 표현하는 방법이 존재한다.
# RGB로 표현 하는 방법이 좀 더 직관적이며 정확하게 명시할 수 있다.(비색 == RGB(170,33,22), 비색을 몰라도 이 색이 붉은 계층임을 알 수 있음)
# 그럼 단어도 벡터로 표현할 수 있지 않을까?
# 분산표현(distrbut;ional representation) : 단어의 의미를 정확학게 파악할 수 있는 벡터 표현
# 분산표현은 고정 길이의 밀지벡터를 사용한다.([0.21, -0.45, 0.83]), 단어의 분산표현을 어떻게 구축할 것인지가 포인트


### 분포 가설
# 자연어 처리의 주요 기법은 모두 단 하나의 간단한 아이디어에 뿌리를 두고 있다. => 분포가설
# 분포가설 : 단어의 의미는 주변 단어에 의해 형성된다.
# 즉, 단어 자체로는 의미가 없고 그 단어가 사용된 맥락이 의미를 형성한다.
# ex) I drink beer, We drin wine ==> drink 주변에는 음료가 등장하기 쉽다.
# ex) I guzzle beer, We guzzle wine ==> guzzle과 drink는 같은 맥락에서 사용되구나!(guzzle과 drink는 가까운 의미 단어이구나!)
# 맥락 : 특정 단어를 중심을 둔 그 주변단어(윈도우 크기에 따라 달라짐)
# ex) You say goodbye and i say hello. ==> goodbye가 중심일 때 window size = 1, say and가 맥락에 포함
# 여기서는 좌우로 똑같은 수의 단어를 맥락으로 사용했지만 상황에 따라 왼쪽 단어만 혹은 오른쪽 단어만 사용하기도 하며 문장의 시작과 끝을 고려하기도 한다.
# 이 책에서는 쉬운 이해를 위해 좌우 동수인 맥락만을 취급


### 동시발생 행렬
# 이제부턴 단어를 벡터로 나타내는 방법을 생각해보자.

# 통계 기반 기법(주목 단어 주위에 어떤 단어가 몇 번이나 등장했는가)
import sys
sys.path.append('...')
import numpy as np
from common.util import preporcess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preporcess(text)

print(corpus) #[0 1 2 3 4 5 6]
print(id_to_word)
# {0 : 'you', 1: 'say', 2:'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

# 윈도우 크기가 1이고 ID가 0 인 you부터 통계기반기법을 사용한다면

#       you     say     goodbye     and     i       hello       .
# you   0       1       0           0       0       0           0
# you라는 단어를 [0,1,0,0,0,0,0] 라는 벡터로 표현할 수 있다.

# 해당 작업을 모든 단어 적용한다면 =>
#           you     say     goodbye     and     i       hello       .
# you       0       1       0           0       0       0           0
# say       1       0       1           0       1       1           0
# goodbye   0       1       0           1       0       0           0
# and       0       0       1           0       1       0           0
# i         0       1       0           1       0       0           0
# hello     0       1       0           0       0       0           1
# .         0       0       0           0       0       1           0

# 이 표의 각 행은 해당 단어를 표현하는 벡터가 된다.
# 위의 표를 '동시발생 행렬(co-occurrence matrix)'라고 한다.

# 동시발생행렬 이용
# C가 동시발생행렬일 때(수기로 적음)
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
],dtype=np.int32)

# 동시발생 행렬을 사용하면 단어의 벡터를 쉽게 얻을 수 있다.
print(C[0]) #ID가 0인 단어의 벡터 표현
#[0 1 0 0 0 0 0]
print(C[4]) #ID가 4인 단어의 벡터 표현
#[0 1 0 1 0 0 0]
print(C[word_to_id['goodbye']]) #"goodbye"의 벡터 표현
#[0 1 0 1 0 0 0]

# 말뭉치로부터 동시발생 행렬을 만들어주는 함수 구현
def creat_co_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id][left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id][right_word_id] += 1

    return co_matrix
        

# 벡터 간 유사도
# 앞에서는 동시발생 행렬을 활용해 단어를 벡터로 표현하는 방법을 알아봤따 이제부턴 벡터 사이의 유사도를 측정하는 방법을 알아볼 것이다!
# 벡터의 유사돌르 나타낼 때는 '코사인 유사도'를 자주 이용한다.
# x dot y = ||x|| ||y|| cos(theta) 에서
# simlarity(x,y) = (x dot y) / (||x|| ||y||) # 이 식의 핵심은 벡터를 정규화하고 내적을 구하는 것.

# 구현
# x,y는 넘파이 배열
def cos_similarity(x,y):
    nx = x / np.sqrt(np.sum(x**2)) 
    ny = y / np.sqrt(np.sum(y**2))
    return np.dot(nx,ny)
# 위의 코드는 x또는y가 0벡터이면 분모가 0이 되어 문제가 생긴다 
# eps(=0.00000001)를 분모에 더해줘서 해결!
def cos_similarity(x,y):
    nx = x / (np.sqrt(np.sum(x**2))+eps)
    ny = y / (np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx,ny)

# 실전예시
# you와 i의 유사도를 구하는 코드
import sys
sys.path.append('...')
from common.util import preprocess, create_co_matrix, cos_similarity
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preporcess(text)
vocab_size = len(word_to_id)
C = creat_co_matrix(corpus,vocab_size)
c0 = C[id_to_word['you']] # 통계기반기법을 거친 you의 벡터
c1 = C[id_to_word['I']] # 통계기반기법을 거친 I의 벡터
print(cos_similarity(c0,c1)) # 0.7071067691154799 #cos_simlar은 -1~1 사이의 값을 가지므로 비교적 유사도가 높다고 할 수 있다.


### 유사 단어의 랭킹 표시
# 단어가 주어졌을 때 해당 단어와 비슷한 단어를 유사도 순으로 출력하는 함수 구현

# 파라미터 정리
# query 검색어(단어), word_to_id, id_to_word(), word_matrix(단어 벡터들을 모은 행렬), top(상위 몇 개까지 출력할지 설정)

def most_similar(query,word_to_id,id_to_word,word_matrix,top=5):
    if query not in word_to_id:
        print(f'{query}를 찾을 수 없습니다.')
        return
    
    print(f'\n[query]'+query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)

    count = 0
    for i in (-1*similarity).argsort():
    # argsort() 메서드 == 오름차순으로 정렬
    # 이때 sdimilarity에 -1를 곱하고 오름차순(크기를 역으로 하고 오름차순) => 내림차순 정렬
    # argsort()의 반환값은 크기별 정렬된 값들의 index 리스트이다.
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word} : {similarity[i]}')

        count += 1
        if count>=top:
            return


### 실제적용
# import sys
# sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch\\')
# from common.util import preprocess, create_co_matrix, most_similar

# text = "You say goodbye and I say hello."
# corpus,word_to_id,id_to_word = preprocess(text)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)

# print(most_similar('you',word_to_id,id_to_word,C,top=5))

# [query] you
#  goodbye: 0.7071067691154799
#  i: 0.7071067691154799
#  hello: 0.7071067691154799
#  say: 0.0
#  and: 0.0
        
# you와 i가 인칭 대명사로 둘이 비슷한것은 이해가 가지만 you와 goodbye,hello의 유사도가 높은 것은 이상하다.
# 물론 지금은 말뭉치의 크기가 너무 작다는 것이 원인이긴 하다.



