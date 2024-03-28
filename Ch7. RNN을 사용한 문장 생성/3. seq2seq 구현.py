### seq2seq 구현 ###

### Encoder 클래스

# 단어 하나하나를 각 계층에서 받고 시계열로 데이터를 처리 및 진행시켜서 최종 h를 출력
# Embedding을 거치고 LSTM에 들어가게 되는데 입력값은 단어의 ID이기 때문에 ID -> 단어 벡터의 변환을 Embedding이 처리한다.
# 각 계층의 출력값은 오른쪽과 위 두 곳으로 전달되게 되는데 위에 계층이 없는 경우 위쪽 출력은 폐기한다.


# cf)
import numpy as np

W1 = np.random.randn(4,5)
W2 = np.random.randn(2,4)


print(W1)
print(W2)

W = [W1,W2]
print(W)

print(W[0][2][0])

# [[ 1.89350214  0.38308002  1.01268208 -0.64754124 -0.24464912]
#  [ 0.48082456  1.07901565 -0.10984122  0.61391771  0.26979739]
#  [-0.33106724 -1.50904276  1.57686176 -0.43375192  0.57791581]
#  [-0.33784308 -0.1091887   0.18119829 -0.478846    0.36334227]]
# [[-1.67041253  0.29098246 -0.34310912  1.81326826]
#  [ 1.17577719 -0.18031258 -0.05906765  0.83739852]]
# [array([[ 1.89350214,  0.38308002,  1.01268208, -0.64754124, -0.24464912],
#        [ 0.48082456,  1.07901565, -0.10984122,  0.61391771,  0.26979739],
#        [-0.33106724, -1.50904276,  1.57686176, -0.43375192,  0.57791581],
#        [-0.33784308, -0.1091887 ,  0.18119829, -0.478846  ,  0.36334227]]), array([[-1.67041253,  0.29098246, -0.34310912,  1.81326826],
#        [ 1.17577719, -0.18031258, -0.05906765,  0.83739852]])]
# -0.3310672444710591

import numpy as np

class Encoder:
    def __init__ (self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx = (rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wh,lstm_Wx,lstm_b,stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.gards + self.lstm.grads
        self.hs = None

    def forward(self,xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:-1:]
    
    def backward(self,dh):
        dhs = np.zeros_like(self.hs)
        dhs[:-1:] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

        
class Decoder:
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx = (rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
        self.affine = TimeAffine(affine_W,affine_b)

        self.params,self.grads = [], []
        for layer in (self.embed,self.lstm,self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self,xs,h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)

        return score
    
    def backward(self,dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
    
    def generate(self,h,start_id,sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1,1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled