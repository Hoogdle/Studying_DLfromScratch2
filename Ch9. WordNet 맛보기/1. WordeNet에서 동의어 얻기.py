### WordeNet에서 동의어 얻기 ###

# NLTK 라이브러리는 자연어 처리를 위한 여러가지 기능 제공(품사 태깅, 구문 분석, 정보 추출, 의미 분석)
# import ntlk로 사용

import nltk
from nltk.corpus import wordnet

print(wordnet.synsets('car'))
# [Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'), Synset('cable_car.n.01')]
# 리스트 요소 5개 => car라는 단어에는 다섯 가지 의미가 정의되어 있다는 뜻
# car.n.01 
# car == 단어이름
# n == 속성(명사,동사 등)
# 01 == 그룹 인덱스

car = wordnet.synset('car.n.01') # 동의 그룹 가져오기
print(car.definition()) # definition == 단어를 이해하고 싶을 때 이용
# a motor vehicle with four wheels; usually propelled by an internal combustion engine
car2 = wordnet.synset('car.n.02')
print(car2.definition())
# a wheeled vehicle adapted to the rails of railroad

print(car.lemma_names()) # car.n.01 동의어 그룹에 어떤 단어들이 존재하는지 확인
# ['car', 'auto', 'automobile', 'machine', 'motorcar']


