### LSTM 구현 ###

# LSTM 에서 수행하는 계산을 정리하면 다음과 같다.
# f = σ(x.t*Wx.f + h.t-1*Wh.f + b.f)
# g = tanh(x.t*Wx.g + h.t-1*Wh.g + b.g)
# i = σ(x.t*Wx.i + h.t-1*Wh.i + b.i)
# o = σ(x.t*Wx.o + h.t-1*Wh.o + b.o)
# c.t = f ⊙ c.t-1 + g ⊙ i
# h.t = o ⊙ tanh(c.t)

# 우리는 가중치 연산을 할 때 각 가중치를 하나의 가중치 W.x에 묶어서 연산할 것이다.
# Wx = [Wx.f, Wx.h, Wx.i, Wx.o]
# Wh = [Wh.f, Wh.h,.Wh.i, Wh.o]
# b = [b.f, b.g, b.i, b.o]
# 처음 4개분의 연산을 한 꺼번에 수행하고 slice를 통해 각각의 결과를 균등하게 나눠준다

import numpy as np

class LSTM:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None

    def forward(self,x,h_prev,c_prev):
        Wx,Wh,b = self.params
        N, H = h_prev.shape

        A = np.matmul(x,Wx) + np.matmul(h_prev,Wh) + b

        #slice
        f = A[:,:H]
        g = A[:,H:2*H]
        i = A[:,2*H:3*H]
        o = A[:,3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g*i
        h_next = o * np.tanh(c_next)

        self.cache = (x,h_prev,c_prev,i,f,g,o,c_next)
        return h_next, c_next
        
    # slice노드의 순전파는 행렬을 네 조각으로 나눠서 분배했다
    # 따라서 slice노드의 역전파는 4개의 기울기를 결합하면 된다
    # np.hstack(), 인수로 주어진 배열들을 가로로 연결한다.
    def backward(self,df,dg,di,do):
        dA = np.hstack((df,dg,di,do))
        return dA
    
### Time LSTM 구현 

class TimeLSTM:
    def __init__(self,Wx,Wh,b,stateful = False):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None,None
        self.dh = None
        self.stateful = stateful

    def forward(self,xs):
        Wx,Wh,b = self.params
        N,T,D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N,T,H), dtype='f') 

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')
        if not self.stateful or self.h is None:
            self.c = np.zeros((N,H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h,self.c = layer.forward(xs[:t:],self.h,self.c)
            hs[:,t,:] = self.h

            self.layers.append(layer)
        
        return hs
    
    def backward(self,dhs):
        Wx,Wh,b = self.params
        N,T,H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N,T,D), dtype='f')
        dh, dc = 0,0

        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:,t,:]+dh,dc)
            dxs[:,t,:] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
            self.dh = dh
            return dxs
        
        def set_state(self,h,c=None):
            self.h,self.c = h,c
        
        def reset_state(self):
            self.h, self.c = None,None
        
        