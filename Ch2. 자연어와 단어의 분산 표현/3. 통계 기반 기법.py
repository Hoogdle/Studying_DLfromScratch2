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