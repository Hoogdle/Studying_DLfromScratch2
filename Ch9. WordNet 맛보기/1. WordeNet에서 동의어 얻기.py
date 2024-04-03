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


### WordNet과 단어 네트워크

# car의 단어 네트워크를 사용하여 다른 단어와의 의미적인 상하 관계를 살펴보기, hypernym_paths() (hypernym == 상위어)
print(car.hypernym_paths()[0])
['car', 'auto', 'automobile', 'machine', 'motorcar']
# [Synset('entity.n.01'), Synset('physical_entity.n.01'), Synset('object.n.01'), Synset('whole.n.02'), Synset('artifact.n.01'), Synset('instrumentality.n.03'), Synset('container.n.01'), Synset('wheeled_vehicle.n.01'), Synset('self-propelled_vehicle.n.01'), Synset('motor_vehicle.n.01'), Synset('car.n.01')]
# entity -> physical_entity -> object -> whole -> artifact -> istrumentality -> container -> wheeled_vehicle -> self-propelled_vehicle -> motor_vehicle -> car
# 위로 갈수록 추상적인, 아래로 갈수록 구체적인 단어가 배치됨

### WordNet을 사용한 의미 유사도
# WordNet에서는 많은 단어가 동의어(유의어) 별로 그룹핑 되어져 있고 단어 사이의 의미 네트워크도 구축돼 있다.
# 이러한 단어 사이의 연결 정보는 다양한 문제에 활용될 수 있다.

# 단어 유사도 계산, path_similarity(), 유사도에 따라 0~1 범위의 실수 반환

car = wordnet.synset('car.n.01')
novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
bus = wordnet.synset('bus.n.01')
cat = wordnet.synset('cat.n.01')

print(car.path_similarity(novel)) #0.05555555555555555
print(car.path_similarity(dog)) #0.07692307692307693
print(car.path_similarity(motorcycle)) #0.3333333333333333

print(bus.path_similarity(car)) #0.125
print(bus.path_similarity(motorcycle)) #0.125

print(cat.path_similarity(dog)) #0.2
print(cat.path_similarity(novel)) #0.047619047619047616

# car은 motorcycle과 가장 유사하고 다음은 dog 다음은 novel 하고 유사한것을 확인할 수 있음.
# car과 motorcycle은 동일한 의미적 상하관계를 다수 공유하지만 novel하고는 거의 공유하지 않는다. => 유사도!