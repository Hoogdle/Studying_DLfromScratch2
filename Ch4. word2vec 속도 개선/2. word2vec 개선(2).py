### word2vec ����(2) ###
# ������ ������ ó��(��İ��� Softmax)
# �װ�Ƽ�� ���ø� ����� ����Ͽ� �ذ�!
# Softmax ��� �װ�Ƽ�� ���ø��� ����ϸ� ���ְ� �ƹ��� ���Ƶ� ��귮�� ���� ���ؿ��� �����ϰ� ������ �� �ִ�.


### ���� �з����� ���� �з���
# ���� �з��� ���� �з��� �ٻ��ϴ� ���� �װ�Ƽ�� ���ø��� key point!
# before
# Q. �ƶ��� you�� goodbye �� �� Ÿ�� �ܾ�� �����Դϱ�?
# A. Ȯ���� say�Դϴ�!
# Now
# Q. �ƶ��� you�� goodbye �� �� Ÿ�� �ܾ say �Դϱ�?
# A. Yes or No
# �ܾ 100���� �ְ� W(in)�� 100�� x 100�̶�� W(out)�� �ڵ����� 100x100�� �̴�.
# ������ ���� �з��� ���ؼ� W(out)�� 100x1�� �Ѵ�! (say�� �ش��ϴ� ���� ����!)
# ���� sigmoid �� ���� Ȯ���� ��´�!

# �Է��� 100�� ������ 100 ����� 100�� �� ��
# Embedding�� ȿ���� �������� ���� ������� 1x100�� ����(�ܾ� say�� ����)�� �ȴ�.
# W(out) (100 x 100��) ���� say�� �ش��ϴ� �ุ�� �����Ѵ� => 100x1 ����
# ���� �� ���͸� ���ϸ� say�� �ش��ϴ� ������ ������ �� �ִ�!


### �ñ׸��̵� �Լ��� ���� ��Ʈ���� ����
# ���� �з� ������ Ǯ���� ������ �ñ׸��̵� �Լ��� Ȯ���� ����� �ս��Լ��ν� '���� ��Ʈ���� ����'�� ����Ѵ�.(���� ���ϰ� ����)
# ���⼭�� loss�� ������ ����.
# L = -(tlogy + (1-t)log(1-y)) (t�� ���� ���̺�, y�� �ñ׸��̵� �Լ��� ���)

# sigmoid�� cross entrophy�� ��ġ�� cal graph���� Loss�� L�̶� ���� ��  ��L/��x = y-t�̴�.
# y�� �ñ׸��̵��� ��°�(Ȯ��) t�� ���� ���̺�� �� ���̰� Ŀ���� ����ġ�� �� ���� �����ϰ� (ũ�� �н�) ���̰� �۾����� ����ġ�� �� ���� �����Ѵ�(�۰� �н�)



### ���� �з����� ���� �з���(����)
# �������� ���信 �ش��ϴ� W(out) index�� Embedding Dot
class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W) #Embedding Ŭ���� ���
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forwrad(self,h,idx):
        target_W = self.embed.forward(idx) # �̴Ϲ�ġ�� ���, idx�� ����Ʈ ������ ���� ����
        out = np.sum(target_W * h,axis=1) # �������� * ������ ���Һ� ���� �����Ѵ�.

        self.cache = (h,target_W) # backward���� ����� �� �ְ� h�� target_W �����ϱ�
        return out # target_W�� ���� numpy �迭�̶�� ���� numpy �迭�� ���ϵ�
    
    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h # �������� * ������ ���Һ� ���� �����Ѵ�.
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    

### �װ�Ƽ�� ���ø�
# ���� ������ ���� ������ ���� ���� �������.
# Q. �ƶ��� you�� gddobye �� �� Ÿ�� �ܾ say �Դϱ�?
# A. Yes or No
# �� ���� you�� goodbye�� �־����� �� say�� Ȯ���� ���̴� ���� ������ you�� goodbye�� �־����� �� �ٸ� �ܾ�(ex)hello)���Դ� ��� �г�Ƽ�� ���� ���մϴ�.
# �� ���� �ʿ��� ���� Ÿ�� �ܾ �ƴ� �ܾ ���� Ȯ���� ���̴� tool�� �ʿ��մϴ�.

# cf) ���� �з������� ���� ������ �ٷ� �� �������� ����� ���� ������ ���� �ٸ��� �з��� �� �־�� �Ѵ�. ���ݱ��� �� �۾��� ���丸�� �з��� �۾��� �ߴ�.
# ������ ó���� Ÿ�� �ܾ ���ؼ��� (�� 1��) ó���ϸ� ���. ������ ������ ó�������� Ÿ�� �ܾ ������ ��� �ܾ ó���ؾ� �Ѵ�. => ���귮 too big...
# ���� ��� �ܾ �ƴ� �� ���� ������ ����� ���ø��Ͽ� ����Ѵ�. == �װ�Ƽ�� ���ø�
# �װ�Ƽ�� ���ø� ����� ����(Ÿ��)�� ��� �ս��� ���ϰ� ���ÿ� ������ ���� �� �� ���ø� �Ͽ� �������� ���� ���ؼ��� �ս��� ���Ѵ�.
# �� �� ������ �������� �ս��� ���� ���� ���� �սǷ� �Ѵ�.
    
# ex) You say goodbye and I say hello. Ÿ�� : say, ���ø� : hello, I
# say�� Embedding Dot �� ���� ���� ���̺��� 1�� 
# hello�� I�� Embedding Dot �� ���� ���� ���̺��� 0����
# ����� ����� �ս��� �� �� �Ͽ� �����ս��� �����Ѵ�.
    

### �װ�Ƽ�� ���ø��� ���ø� ���
# �׷��ٸ� � �ܾ���� ���ø��� �ĺ��� �Ѱ��ΰ�? 
# ����ġ���� �ܾ��� ���� �󵵼��� �������� �� �ܾ��� ���� Ȯ�� ������ ���� �� Ȯ�� ������� �װ�Ƽ�� �ܾ ���ø��Ѵ�.
# => ���� ������ �ܾ ���õ� ���ɼ��� ���� ������ ����� �ܾ ���õǱ�� ��ƴ�(good)

# Ȯ�� ������ ���� ���ø��ϴ� �ڵ� ����
import numpy as np

# 0~9 ������ ���� �� �ϳ��� �������� ���ø�
print(np.random.choice(10)) #7

# �ܾ� ����Ʈ���� �������� �ܾ� �ϳ� ���ø�
words = ['you','say','goodbye','I','hello','.']
print(np.random.choice(words)) #hello

# 5���� �������� ���ø�(�ߺ�����) size�� ũ�� ����
print(np.random.choice(words,size=5)) #['hello' 'I' 'you' 'say' 'you']

# 5���� �������� ���ø�(�ߺ�����) size�� ũ�� ����
print(np.random.choice(words,size=5,replace=False)) #['hello' 'say' 'goodbye' 'you' 'I']

# Ȯ�� ������ ���� ���ø�
p = [0.5,0.1,0.05,0.2,0.05,0.1]
print(np.random.choice(words,size=3,p=p)) #['you' 'you' 'you']

print(np.random.choice(words,size=3,replace=False)) #['you' 'say' '.']


# �̶� ������ ���� ���� ���ø��� �����ϰ� �ϱ� ���� �� �ܾ��� ������ 0.75������ ���ش�. => ������ ���� �ܾ��� ������ �ö󰣴�!
# P'(w.i) = (P(w.i)^0.75) / ((j~n).�� (P(w.i)^0.75))

p = [0.7,0.29,0.01]
new_p = np.power(p,0.75)
new_p /= np.sum(new_p)
print(new_p) #[0.64196878 0.33150408 0.02652714]

# �ش� ���翡���� UnigramSampler Ŭ������ �� �ڵ���� ����
# UnigramSampler Ŭ������ �Ű������� �ܾ� ID ����� 'Corpus', Ȯ�������� ������ �� ���� 'power', ���ø� �� �������� ������ 'sample_size'�� �ִ�.\
# get_negative_sample �̶�� �޼��带 ���� target �μ��� �ܾ���� �������� ���� �ν��ϰ� �� ���� �ܾ� ID�� ���ø��Ѵ�.
# ��뿹��
corpus = np.array([0,1,2,3,4,1,2,3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus,power,sample_size)
target = np.array([1,3,0]) # Ÿ����� �̴Ϲ�ġ��!
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)
# [[0 3] # 1�� Ÿ���� ���� sample
#  [1 2] # 3�� Ÿ���� ���� sample
#  [2 3]] # 0�� Ÿ���� ���� sample



### �װ�Ƽ�� ���ø� ����
# NegativeSamplingLoss Ŭ������ ����
class NegativeSamplingLoss:
    def __init__(self,W,corpus,power=0.75,sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus,power,sample_size):
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]
        self.params,self.gards = [],[]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.gards += layer.grads

    def forward(self,h,target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # ������ �� ������
        score = self.embed_dot_layers[0].forward(h,target)
        corret_label = np.ones(batch_size,dtype=np.int32)
        loss = self.loss_layers[0].forward(score,corret_label)

        # ������ �� ������
        negative_label = np.zeros(batch_size,dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h,negative_target)
            loss += self.loss_layers[1+i].forward(score,negative_label)

        return loss
    
    def backward(self,dout=1):
        dh = 0
        for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh

### CBOW �� ����
    
