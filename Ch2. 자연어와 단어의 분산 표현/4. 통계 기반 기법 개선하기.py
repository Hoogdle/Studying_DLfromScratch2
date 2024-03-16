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