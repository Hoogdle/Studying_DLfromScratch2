### GRU의 인터페이스 ###

# LSTM에서 각 계층으로 이어지는(흘러가는) 선은 은닉상태(h)와 기억상태(c)이다.
# GRU에서는 only 은닉상태 선만을 이어지게 한다.

# cf) LSTM은 필요한 정보를 기억 셀에 기록, 기억 셀의 정보를 바탕으로 은닉 상태를 구현
# GRU는 추가 기억 장소를 이용하지 않는다.

# LSTM에서는 Input gate,Forget gate, Output gate를 사용하였다면 
# GRU에서는 Reset gate과 Update gate를 사용한다.
# Reset gate는 은닉 상태를 얼마나 '무시'할지 정한다. r이 0이면 과거의 은닉상태는 완전히 무시된다.
# Reset gate의 값의 크기에 따라 새로운 은닉상태 h를 계산할 때 과거 은닉상태를 얼마나 사용할 것인지 결정된다.(r big, 과거 은닉상태 고려 많이, r small, 과거 은닉상태 고려 적게)

# Update gate, z는 LSTM의 input,output gate를 담당하는 gate로 z값에 따라 과거 은닉상태를 잊고 새로 추가된 은닉 상태에 가중치를 부여한다.

# z = σ(x.t*Wx.z + h.t-1*Wh.z + b.z)
# r = σ(x.t*Wx.r + h.t-1*Wh.r + b.r)
# h.~ = tanh(x.t*W.x + (r⊙h.t-1)W.h + b)
# h.t = (1-z)⊙h.t-1 + z⊙h.~

# 즉, GRU는 LSTM을 더 단순하게 만든 아키텍쳐이다. => 계산 비용과 매개변수 수를 줄임
