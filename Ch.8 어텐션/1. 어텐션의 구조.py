 ### 어텐션의 구조 ###

### seq2seq의 문제점
# seq2seq에서는 Encorder가 시계열 데이터를 인코딩한다. Encorder의 출력은 '고정 길이의 벡터' 이다.
# '고정 길이' 벡터이기 때문에 입력 문장이 아무리 길어도 항강 같은 길이의 벡터로 변환된다.
# ex)
# "나는 고양이로소이다." => [벡터]
# "아무튼 어두컴컴하고 축축한 데서 야옹야옹 울고 있었던 것만은 분명히 기억한다" => [벡터]
# 아무리 긴 문장이라도 고정된 길이의 벡터로 변환하기 때문에 한계가 있다. (필요한 정보를 전부 담지 못한다)



### Encoder 개선
# 지금까지 LSTM 계층의 마지막 은닉 상태만을 Decorder에게 전달 하였다(not good)
# Enocder의 출력 길이는 입력 문장의 길이에 따라 바꿔주는 것이 좋다.
# 즉, LSTM 계층의 은닉 상태 벡터를 모두 사용하는 것이다.
# 은닉 상태 벡터를 모두 이용하게 되면 입력된 단어의 갯수만큼 Encorder가 벡터를 출력하게 된다.즉, 하나의 고정된 길이 벡터에서 해방되게 된다.
# 시각별 LSTM 계층의 은닉 상태에는 직전에 입력된 단어에 댛단 정보가 많이 포함되어 있다.
# ex) "고양이" 라는 단어가 입력되었을 때의 LSTM 계층의 출력은 직전에 입력한 "고양이" 라는 단어의 영향을 가장 크게 받는다.
# 즉 Encoder가 출력하는 hs 행렬은 각 단어에 해당하는 벡터들의 집합이라고 볼 수 있다.
# 이로써 Encorder는 입력 단어에 비례하는 정보를 인코딩 할 수 있게 되었다.




### Decoder 개선

