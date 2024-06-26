### 어텐션에 관한 남은 이야기 ###

### 양방향 RNN
# (seq2seq의 Encoder에 초점을 맞춤.)
# Encoder의 입력값으로 "나는 고양이로소이다." 가 들어간다면 고양이의 은닉 상태에는 '나','는','고양이'의 정보가 들어가게 된다.
# 전체적인 균형을 생각해보면 고양이 단어의 '주변' 정보를 균형 있게 담고 싶다.
# => LSTM을 양방향으로 처리 == 양방향 LSTM(양방향 RNN)

# 양방향 LSTM은 기존의 순방향 LSTM에다가 역방향 LSTM을 추가한 형태이다.
# 양방향 LSTM의 출력값은 순방향 LSTM의 출력값(은닉상태)와 역방향 LSTM의 출력값(은닉상태)를 연결(또는 합, 또는 평균)한 벡터이다.
# => 각 단어에 대응하는 은닉 상태 벡터에는 좌와 우 양쪽 방향으로부터의 정보를 집약할 수 있다. => 균형 잡힌 정보가 인코딩

# 구현 방법
# 두 개의 LSTM을 사용하게 된다. 하나는 순방향 다른 하나는 역방향
# 역방향 LSTM은 input으로 주는 단어를 역순으로 주면 된다.
# ex) 입력 단어 시계열이 A,B,C,D 일때
# 순방향의 입력 : A,B,C,D
# 역방향의 입력 : D,C,B,A
# 이후 각 LSTM의 출력값을 연결하면 된다.


### Attention 계층 사용 방법
# 지금까지 우리는 Attention을 Affine 계층 직전에 Attention 계층을 만들어 Attention의 출력과 LSTM의 출력을 연결하여 Affine 계층의 입력으로써 사용하였다.
# 위와 같은 방법으로 해야하는 것은 절대 아니며 Attention 계층을 자유롭게 사용해도 된다.(ex, LSTM 계층에 사용)
# 어떤 결과가 좋은지는 학습을 해보기전에는 아무도 모른다. 즉 실제 데이터로 검증을 해야한다.


### seq2seq 심층화와 skip 연결
# 지금 까지 구현한 Attention LSTM은 LSTM의 계층이 단 1개였지만 LSTM을 수직방향으로 깊게 쌓을수록 더 표현력이 높은 모델을 만들 수 있다.
# Encoder와 Decoder가 같은 깊이의(층의) LSTM을 사용하는 것이 일반적이다.
# Attention 계층의 사용법은 여러 변형이 존재한다!!!

# Residual Connection
# shor-cut 또는 skip으로 부르기도 한다.
# 이전 계층의 출력값을 현재 계층의 출력값에다가 더해주는 연산
# '더해주는 연산'이 핵심! 연산 그래프를 떠올려보자! 덧셈 연산은 기울기를 그대로 흘려준다.
# A라는 계층이 LSTM이 2개가 쌓여 있다고 하자. A의 입력값이 x일 때 residual connection을 적용하면 출력값은 다음과 같다.
# x + A(x), 역전파를 하게 되면 A이전 계층에는 A 이후 계층의 역전파 흐름이 그대로 전달되게 된다.(덧셈 연산 이기 때문)
# 즉, A 계층의 역전파를 전체 거치면서 발생하는 기울기 손실(LSTM의 tanh함수)을 A이후 계층의 기울기를 A이전 계층에게 그대로 전달함으로써 기울기 손실을 방지할 수 있다.
# RNN 계층의 역전파에서는 시간(수평) 방향과 수직 방향에서 기울기 소실 또는 기울기 폭발이 일어날 수 있다.
# 수평 방향에서의 기울기 소실은 '게이트가 달린 RNN'으로 처리하고 기울기 폭발은 '기울기 클리핑'으로 대응한다
# 수직 방향에서의 기울기 소실은 skip 연결로 대응한다.