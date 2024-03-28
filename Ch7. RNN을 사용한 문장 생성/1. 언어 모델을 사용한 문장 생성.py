### ��� ���� ����� ���� ���� ###
# ��� �𵨷� ������ �����ϱ�!

### RNN�� ����� ���� ������ ����
# ex) You say goodbye and I say hello. ��� �������� �н��� RNN ���� �ִٰ� ��������
# �� �𵨿��� 'I'��� �Է� ���� �ָ� �� �ܾ��� Ȯ�� �������� say�� Ȯ���� ���� ���� ���̴�.
# Ȯ���� ���� ���� �ܾ �����ϴ� '�������� ���' ���ٴ� Ȯ���� ���� �ܾ��ϼ��� ���õ� Ȯ���� ����, ���� �ܾ��ϼ��� ���õ� Ȯ���� ���� 'Ȯ���� ����' ����� ����Ѵ�(�� å����)
# I�� ��°����� say�� ���� ���̰� say�� ��°����� hello �� ���� ���̴�.(���� Ȯ���� ����.)
# ���⼭ ������ ������ �Ʒ� �����Ϳ��� �������� �ʴ�, ���� ������ �����̴�.(��� ���� �Ʒ� �����͸� �ϱ��ϴ� ���� �ƴ� ���� �ܾ��� ���� ������ �н��ϱ� ����)


### ���� ���� ����

import sys
sys.path.append('...')
import numpy as np
from common.fucntions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self,start_id,skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids)<sample_size:
            x = np.array(x).reshape(1,1) # RNN�� input�� ��� 2������ ����, ���� x�� 2�������� ��������
            score = self.predict(x)
            p = softmax(score.flatten())
            
            sampled = np.random.choice(len(p),size=1,p=p) #len(p)�� ���� word�� id�� ���õ�. p�� Ȯ�� ����Ʈ
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids
    

### RNNlmGen Ŭ������ ���� �����ϱ�
import sys
sys.path.append('...')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N','<unk>','$']
skip_ids = [word_to_id[w] for w in skip_words]

word_ids = model.generate(start_id,skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>','\n')
print(txt)



