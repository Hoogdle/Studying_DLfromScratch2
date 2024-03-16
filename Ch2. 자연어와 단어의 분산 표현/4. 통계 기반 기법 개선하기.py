### ��� ��� ��� �����ϱ� ###

### ��ȣ ������
# ���ù߻� ����� ���Ҵ� �� �ܾ ���ÿ� �߻��� Ƚ���� ��Ÿ����.
# the car �� drive car ����
# car�� drive�� �� ������ �־� �������� the car�̶�� '������ ��'�� �� ���� ������ car�� drive�� �ƴ� the�� �� �����ϴٰ� �Ǵ��Ѵ�(������)
# => ���� ��ȣ������(Pointwise Mutual Information - PMI) ô���� �� ���� �ذ�!
# PMI ô���� �� �ܾ ���ÿ� �߻��� Ȯ���� ���(����)�ϵ�, �� �ܾ ��Ÿ�� Ȯ���� ���(�и𿡼�)�Ͽ� ���� ������ �ذ��Ѵ�.
# �� car�� drive���� �󵵼��� ���� the car�� ��Ʈ�� drive car�� ��Ʈ���� ���� �� �ִ��� car�� �󵵼��� drive�� �󵵼��� �и�ν� ������� ���� ���� drive car�� �� ���� �� �ִ�.

# PIM(x,y) = log2(P(x,y)/P(x)P(y)) = log2((C(x,y)/N)/((C(x)/N)(C(y)/N))) = log2((C(x,y)*N)/(C(x)C(y)))
# log2 : ���� 2�� �α׸� ����ٴ� �� / P(x,y) x��y �ܾ ���ÿ� �߻��� Ȯ�� / C(x,y) x��y�� ���ÿ� �߻��� ����

# ���� �� �ܾ ���ÿ� �߻��ϴ� ��찡 �������� ������ log2(0)���� -�İ� �ȴ�.
# => ���� ��ȣ������(Positive PMI - PPMI)�� �ذ�!
# PPMI(x,y) = max(0,PMI(x,y)) #PMI(x,y)�� ���� 0���� ������ 0���� ��ȯ, 0���� ũ�� �״�� ��ȯ

### ppmi ����

# C ���ù߻� ���, verbose �����Ȳ ��� ����, np.log2(0)�� �Ǵ� ���� �������� eps
# ���ù߻� ��Ŀ� ���ؼ��� PPMI ����� ���� �� �ֵ��� �ܼ�ȭ�� �ڵ�(�ٻ��� ���ϵ���)
# => C(x) = i.�� C(i,x) / C(y) = i.�� C(i,y) / N = i.j.�� C(i,j)
# Original
# N = corpus �ܾ��� ��, C(x) = x�� �߻��� �� / N, C(y) = y�� �߻��� �� / N
import numpy as np

def ppmi(C,verbose=False,eps=1e-8): 
    M = np.zeros_like(C,dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0) # �Ʒ��������� ��� add (�� �ܾ��� �󵵼� ���ϱ�)
    total = C.shape[0]*C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[i]*S[j]) + eps)
            M[i,j] = max(0,pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print(f'{100*cnt/total : .1f}%%�Ϸ�')
    return M


### ���� ����
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3) # ��ȿ �ڸ����� �� �ڸ��� ǥ��
print('���ù߻� ���')
print(C)
print('-'*50)
print('PPMI')
print(W)

# ���ù߻� ���
# [[0 1 0 0 0 0 0]
#  [1 0 1 0 1 1 0]
#  [0 1 0 1 0 0 0]
#  [0 0 1 0 1 0 0]
#  [0 1 0 1 0 0 0]
#  [0 1 0 0 0 0 1]
#  [0 0 0 0 0 1 0]]
# --------------------------------------------------
# PPMI
# [[0.    1.807 0.    0.    0.    0.    0.   ]
#  [1.807 0.    0.807 0.    0.807 0.807 0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.    1.807 0.    1.807 0.    0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.807 0.    0.    0.    0.    2.807]
#  [0.    0.    0.    0.    0.    2.807 0.   ]]

### PPMI�� ������
# ����ġ�� ���� ���� �����Կ� ���� �� �ܾ��� ������ ���� ���� ����.(����ġ ���ּ��� 10�� => ���� 10�� ����....what!)
# PPMI�� ����� ���� �� ��κ� 0, �� ��κ��� �߿����� �ʴµ� ����. ==> ���� ���� �ʿ�!



### ���� ����(Dimensionality Reduction)
# �߿��� ������ �����ϸ鼭 ������ ������ ���̴� ���

# �������� ����
# 1. 2������ �����Ͱ� ������ �� �������� ����(���� ���)�� ���� ���� ������ ���� �ϳ� �����Ѵ�
# 2. �ش� ������ ��� �����͸� ���翵 �Ѵ�. �� �������� ���� ���ο� ������ ���翵�� ������ ����ȴ�.
# ������ ���� ã�Ƴ��� ���� ���� �߿��ϴ�!

# cf) ������, ��Һ���
# : ��κ��� 0�� ���, ����
# ���� ������ �ٽ��� ��Һ��Ϳ��� �߿��� ���� ã�Ƴ��� �� ���� �������� �ٽ� ǥ���ϴ� ��
# ���� ����(���翵)�� ����� ���� ��κ��� 0�� �ƴ� ������ ������ '��������'�� ��ȯ
# �� ������ ���;߸��� �츮�� ���ϴ� �ܾ��� �л� ǥ��

# Ư�հ�����(Singular Value Decompositon - SVD)
# ������ ���ҽ�Ű�� ��� �� �ϳ�
# SVD �� ������ ����� �� ����� ������ �����Ѵ�.
# X = USV.T # V transpose # U�� V�� �������(�� ���ʹ� ���� ����) # S�� �밢��ķ� �밢 ���� �ܿ��� ��� 0�� ���
# X �� �ܾ���� matrix (�� ���� �� �ܾ��� ���͵�)
# U�� ������� �̹Ƿ� � ������ ��(����)�� �����Ѵ�. => U�� '�ܾ� ����'���� ����� �� ����
# S�� �밢��ķ� �밢���п��� 'Ư�հ�'�� ū ������� �����Ǿ� �ִ�. Ư�հ��� '�ش� ��'�� �߿䵵��� ������ �� �ִ�.

# ���� ������ ������ ������ ���� (Ű ����Ʈ�� S!)
# S�� �밢������ ū ������ �����Ǿ� �ִ�. �� �� �߿����� ����(�޺κ�) ������ ������ => S'
# S'�� ���ϰ� �Ǹ� ����� ���� ���߱� ���� U�� Vt�� ���� �� �ۿ� ���� => U' Vt'
# �̷��� �Ǹ� U�� ������(�ܾ��� ���� ����)�� �ҽǵǰ� �Ǵ°�! == ��������


### SVD �������� ����
# SVD�� �������� linalg ����� ����� ���� ����(linalg�� ��������� ���)

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

# SVD
U,S,V = np.linalg.svd(W)

print(C[0]) #[0 1 0 0 0 0 0]
print(W[0]) #[0.        1.8073549 0.        0.        0.        0.        0.       ]
print(U[0]) #SVD ��ü, �������ͷ� ��ȯ��. #[-1.1102230e-16  3.4094876e-01 -1.2051624e-01 -3.8857806e-16  0.0000000e+00 -9.3232495e-01  8.7683712e-17]

# ������ ������ ���ҽ�Ű���� just ���ϴ� ��ŭ�� ������ ��.
print(U[0][:2]) #[-1.1102230e-16  3.4094876e-01] # 2���� ����

# �׷����� ��Ÿ����
for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id,0],U[word_id,1])) #x,y ����(����, U[word_id,0]�� U[word_id,1]�� ��Ÿ���� ����)�� word�� ��� �ؽ�Ʈ�� �׸����� ��

plt.scatter(U[:,0],U[:,1],alpha=0.5) # U ��� �� (��ü�� and 0��)�� (��ä�� and 1��)�� �׷����� ǥ����. ������ 0.5
plt.show() # �׷��� ��Ÿ��

# ������ �����ͼ��� �۾Ƽ� �մ����� ���� ����� �������� �����δ� PTB��� ū ����ġ�� ����Ͽ� ���� ����


### PTB(Penn Treebank) �����ͼ�
# PTB�� ����ġ�� �ϳ�
# ����� �ܾ�� <unk>�� ġȯ (unknown)
# ��ü���� ���ڴ� N���� ġȯ
# <eos> ������ ��


# PTB ���
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train') #train,valid,test�� ���� �� �� ����

print('����ġ�� ũ�� :',len(corpus))
print(f'corpus[:30] : {corpus[:30]}')
print()
print(f'id_to_word[0] : {id_to_word[0]}')
print(f'id_to_word[1] : {id_to_word[1]}')
print(f'id_to_word[2] : {id_to_word[2]}')
print()
print(f'word_to_id["car"] : {word_to_id["car"]}')
print(f'word_to_id["happy"] : {word_to_id["happy"]}')
print(f'word_to_id["lexus"] : {word_to_id["lexus"]}')

# ����ġ�� ũ�� : 929589
# corpus[:30] : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29]

# id_to_word[0] : aer
# id_to_word[1] : banknote
# id_to_word[2] : berlitz

# word_to_id["car"] : 3856
# word_to_id["happy"] : 4428
# word_to_id["lexus"] : 7426



# ptb �����ͷ� ���͸���� ���ϱ�

import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

C = create_co_matrix(corpus,vocab_size,window_size)
W = ppmi(C, verbose=True)

try:
    from sklearn.utils.extmath import randomized_svd #for ���� ����
    U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter=5,random_state=None)
except ImportError:
    U,S,V = np.linalg.svd(W) #������ �� �Ǹ� ��¿ �� ���� ���� ����

word_vecs = U[:,:wordvec_size]

querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query,word_to_id,id_to_word,word_vecs,top=5)


# [query] you
#  i: 0.7016294002532959
#  we: 0.6388040781021118
#  anybody: 0.5868049263954163
#  do: 0.5612815022468567
#  'll: 0.512611985206604

# [query] year
#  month: 0.6957005262374878
#  quarter: 0.691483736038208
#  earlier: 0.6661214232444763
#  last: 0.6327787041664124
#  third: 0.6230477094650269

# [query] car
#  luxury: 0.6767407655715942
#  auto: 0.6339930295944214
#  vehicle: 0.597271203994751
#  cars: 0.5888376235961914
#  truck: 0.5693157911300659

# [query] toyota
#  motor: 0.7481387853622437
#  nissan: 0.7147319912910461
#  motors: 0.6946365833282471
#  lexus: 0.6553674936294556
#  honda: 0.6343469619750977
    

### ����
# ��ǻ�Ϳ��� '�ܾ��� �ǹ�'�� ���ؽ�Ű�� ���� ���� ����Ʈ�� �̹� é�Ͱ� �����.
# �üҷ����� ���� �η�,ǥ������ �Ѱ�... => ��� ��� ���
# ��� ��� ����� ����ġ�κ��� �ܾ��� �ǹ̸� �ڵ����� �����ϰ� �� �ǹ̸� ���ͷ� ǥ��
# (window�� �ܾ� count�� ��������) ���ù߻� ����� ���� => PPMI ��ķ� ��ȯ => SVD�� ������ ����, �ܾ��� �л� ǥ��(����� �ܾ���� ���� ���������� ������ �� ����) => cos ���絵�� ���缺 Ȯ��