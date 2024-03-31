### seq2seq 개선 ###

### 1. 입력 데이터 반전(Reverse)
# just 입력 데이터를 반전 시키는 방법

# "나는 고양이로소이다" => "I am a cat"으로 번역하는 문제
# '나'부터 'I'로 가려면 '고양이', '로소', '이다' 와 같은 단어들을 거치면서 가야한다.(역전파 시 'I'에서 '나'까지 전해지는 기울기는 길이에 영향을 받게 됨)
# "이다 로소 고양이 나는" 으로 변경한다면 
# "이다 로소 고양이 나는" => "I am a cat" 으로 역전파시 길이의 영향을 덜 받게 된다.(하지만 각 단어의 평균적 길이는 동일)

(x_train,t_train),(x_test,t_test) = sequence.load_data('addition.txt')
x_train, x_test = x_train[:,::-1], x_test[:,::-1] # 배열의 열을 반전


### 2. 엿보기(Peeky)
# 기존의 Seq2seq 모델은 Encoder 마지막 출력 h를 Decoder 첫 번째 계층만이 사용하였다.
# Encoder의 마지막 출력 h는 Encoder 에서의 모든 정보를 담고 있는 벡터이므로 모든 계층이 사용하는 것이 좋아 보인다.
# => 모든 계층(LSTM, Affine)에서도 h를 이용하자!!(h를 모든 계층이 엿 본다는 의미로 Peeky)

import numpy as np

class PeekyDecoder:
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx = (rn(H+D,4*H)/np.sqrt(H+D)).astype('f')
        lstm_Wh = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H+H,V)/np.sqrt(H+H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLstm(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
        self.affine = TimeAffine(affine_W,affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed,self.lstm,self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
    
    def forward(self,xs,h):
        N,T = xs.shape
        N,H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h,T,axis=0).reshape(N,T,H)
        out = np.concatenate((hs,out),axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs,out),axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score
    

### PeekySeq2seq 구현
    
from seq2seq import Seq2seq, Encoder

class PeekySeq2seq(Seq2seq):
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = PeekyDecoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = 