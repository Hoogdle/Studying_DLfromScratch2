### RNN 구현 ###

# RNN 계층 : 한 단계 작업을 수행하는 계층
# Time RNN 계층 : T개 단계분(배치사이즈만큼)의 작업을 한꺼번에 처리하는 계층
# 한 단계만을 처리하는 RNN 계층을 구현한 다음 RNN 계층을 이용해 T개 단계의 처리르 한꺼번에 수행하는 TimeRNN 클래스 완성하도록 한다.

### RNN 계층 구현
# RRN 순전파 식:
# h.t = tanh(h.t-1 x W.h + x.t x W.x + b)
import numpy as np
class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None

    def forward(self,x,h_prev):
        Wx,Wh,b = self.params
        t = np.matmul(h_prev,Wh) + np.matmul(x,Wx) + b
        h_next = np.tanh(t)

        self.cache = (x,h_prev,h_next) # 역전파에 사용될 데이터 저장
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b = self.params
        x,h_prev,h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt,axis=0)
        dWh = np.matmul(h_prev.T,dt)
        dh_prev = np.matmul(dt,Wh.T)
        dWx = np.matmul(x.T,dt)
        dx = np.matmul(dt,Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx,dh_prev
    

### Time RNN 계층 구현

class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False): # stateful : 상태가 있는 ==> 은닉 상태를 유지
        # cf) 긴 시계열 데이터를 처리할 때는 RNN의 은닉 상태를 유지해야 한다.
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None

    def set_state(self,h):
        self.h = h
    def reset_state(self,h):
        self.h = None
    
    def forward(self,xs): # 아래로 부터 입력 xs
        Wx,Wh,b =self.params
        N,T,D = xs.shape # N:미니배치 크기 // D : 입력 벡터의 차원의 수 // T : T개 분량의 시계열 데이터
        D,H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H), dtype='f') # 출력을 담을 그릇

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h)
            # 데이터가 시계열인 경우에는 서로다른 미니배치에 있는, 같은 시계열을 공유하는 데이터를 동시에 인풋으로 주는 것이 일반적이다.
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs
    
    def backward(self, dhs):
        Wx, Wh, b =self.params
        N,T,H = dhs.shape
        D,H = Wx.shape

        dxs = np.empty((N,T,D),dtype='f')
        dh = 0
        grads = [0,0,0]
        
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx,dh = layer.backward(dhs[:,t,:]+dh)
            dxs[:,t,:] = dx

            for i,grad in enumerate(layer.grads):
                self.grads[i] = grad
        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs


    