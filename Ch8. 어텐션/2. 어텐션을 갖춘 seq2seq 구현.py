### 어텐션을 갖춘 seq2seq 구현 ###

# AttentionEncoder, AttentionDecoder, AttentionSeq2seq 구현

### Encoder 구현
# 마지막 은닉 벡터를 반환하는 것이 아닌 모든 계층의 은닉 벡터를 반환하는 것이 차이점 

import numpy as np
import sys
sys.path.append('...')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention

class AttentionEncoder(Encoder):
    def forward(self,xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs
    
    def backward(self,dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
    
### Decoder 구현
class AttentionDecoder:
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn
        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx = (rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(2*H,V)/np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
        self.attention =TimeAttention()
        self.affine = TimeAffine(affine_W,affine_b)
        layers = [self.embed,self.lstm,self.attention,self.affine]
        