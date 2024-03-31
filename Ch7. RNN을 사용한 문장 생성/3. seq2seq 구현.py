### seq2seq ���� ###

### Encoder Ŭ����

# �ܾ� �ϳ��ϳ��� �� �������� �ް� �ð迭�� �����͸� ó�� �� ������Ѽ� ���� h�� ���
# Embedding�� ��ġ�� LSTM�� ���� �Ǵµ� �Է°��� �ܾ��� ID�̱� ������ ID -> �ܾ� ������ ��ȯ�� Embedding�� ó���Ѵ�.
# �� ������ ��°��� �����ʰ� �� �� ������ ���޵ǰ� �Ǵµ� ���� ������ ���� ��� ���� ����� ����Ѵ�.


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
    

### Seq2seq Ŭ����
# EncoderŬ������ Decoder Ŭ������ ����
# Time Softamx with Loss �������� �ս� ���

class Seq2seq(BaseModel):
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = Decoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.gards = self.encoder.grads + self.decoder.grads

    def forward(self,xs,ts):
        decoder_xs, decoder_ts = ts[:,:-1], ts[:,1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs,h)
        loss = self.softmax(score,decoder_ts)
        return loss
    
    def backward(self,dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self,xs,start_id,sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h,start_id,sample_size)
        return sampled
    

### seq2seq ��
# seq2seq�� �н��� �⺻���� �Ű���� �н��� ���� �н����� �̷�����.
# 1. �н� �����Ϳ��� �̴Ϲ�ġ�� ����
# 2. �̴Ϲ�ġ�κ��� ���⸦ ���
# 3. ���⸦ ����Ͽ� �Ű������� �����Ѵ�.
import sys
sys.path.append("C:\\Users\\rlaxo\\Desktop\\deepscratch")
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

#  ������ set�� �б�
(x_train,t_train), (x_test,t_test) = sequence.load_data('addtion.txt')
char_to_id, id_to_char = sequence.get_vocab()

# �������Ķ���� ����
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# �� / ��Ƽ������ / Ʈ���̳� ����
model = Seq2seq(vocab_size,wordvec_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train,t_train,max_epoch=1,batch_size=batch_size,max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]],t_test[[i]]
        verbose = i<10
        correct_num += eval_seq2seq(model,question,correct,id_to_char,verbose)
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print('���� ��Ȯ�� %.3f%%' %(acc*100))  


    
