### 시계열 데이터 처리 계층 구현 ###
# 이번 장의 목표는 RNN을 사용하여 '언어 모델'을 구현하는 것
# RNNLM : RNN Language Model


### RNNLM의 전체 그림
# You say goodbye and I say hello.
# 이 문장에서 You라는 단어가 처음으로 들어가게 되고 출력값이 출력층, 그 다음 은닉층으로 전달되게 된다.
# 이때 출력층으로 전달되는 것은 Affine 계층을 거쳐 Sotfmax로 출력하게 되면 다음 단어의 확률(say)이 높아진 것을 알 수 있다.
# 또한 say의 RNN 층의 출력값을 Affine => Softmax 로 보면 goodbye와 hello의 확률이 높은 것을 알 수 있는데 
# 이는 RNN 계층이 "You say"라는 맥락을 기억하고 있다는 것을 반증해줄 수 있다.
# 더 정확하게 말하면 RNN은 "You say"라는 과거의 정보를 응집된 은닉 상태 벡터로 저장해두고 있다.
# RNN 계층은 과거에서 현재로 데이터를 계속 흘려보내줌으로써 과거의 정보를 인코딩해 저장 할 수 있다!