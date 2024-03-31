### seq2seq ���� ###

### 1. �Է� ������ ����(Reverse)
# just �Է� �����͸� ���� ��Ű�� ���

# "���� ����̷μ��̴�" => "I am a cat"���� �����ϴ� ����
# '��'���� 'I'�� ������ '�����', '�μ�', '�̴�' �� ���� �ܾ���� ��ġ�鼭 �����Ѵ�.(������ �� 'I'���� '��'���� �������� ����� ���̿� ������ �ް� ��)
# "�̴� �μ� ����� ����" ���� �����Ѵٸ� 
# "�̴� �μ� ����� ����" => "I am a cat" ���� �����Ľ� ������ ������ �� �ް� �ȴ�.(������ �� �ܾ��� ����� ���̴� ����)

(x_train,t_train),(x_test,t_test) = sequence.load_data('addition.txt')
x_train, x_test = x_train[:,::-1], x_test[:,::-1] # �迭�� ���� ����


### 2. ������(Peeky)
# ������ Seq2seq ���� Encoder ������ ��� h�� Decoder ù ��° �������� ����Ͽ���.
# Encoder�� ������ ��� h�� Encoder ������ ��� ������ ��� �ִ� �����̹Ƿ� ��� ������ ����ϴ� ���� ���� ���δ�.
# => ��� ����(LSTM, Affine)������ h�� �̿�����!!(h�� ��� ������ �� ���ٴ� �ǹ̷� Peeky)

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
    

### PeekySeq2seq ����
    
from seq2seq import Seq2seq, Encoder

class PeekySeq2seq(Seq2seq):
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = PeekyDecoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = 