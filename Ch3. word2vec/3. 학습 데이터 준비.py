### �н� ������ �غ� ###
# word2vec�� ���� �н� ������ �غ�

### �ƶ��� Ÿ��
# word2vec�� �Է��� '�ƶ�'�̰� ���� ���̺��� 'Ÿ��'(�ƶ��ȿ� �� �߾��� �ܾ�)�̴�.
# ������ �ƶ��� �־��� �� Ÿ���� ������ Ȯ���� ���̴� ���̴�.

# �ƶ��� Ÿ���� ����� ��
#                                           �ƶ�            Ÿ��           
# you say goodbye and I say hello.      you,goodbye         say
# you say goodbye and I say hello.      say,and             goodbye
# you say goodbye and I say hello.      goodbye,I           and
# you say goodbye and I say hello.      and,say             I
# you say goodbye and I say hello.      I,hello             say
# you say goodbye and I say hello.      say,.               hello

# �� �۾��� ����ġ���� ���� �� �ܾ �����ϰ� ��� �ܾ ����
# �ƶ��� ���� �������� �� �� ������ Ÿ���� �� �ϳ�!

### �ƶ��� Ÿ���� ����� �ڵ� ����

import numpy as np
import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
print(corpus) #[0 1 2 3 4 1 5 6] 
# cf) corpus�� ����ġ�� word���� id�� ��ȯ�� ����Ʈ(�ߺ����)
# word_to_list�� ����ġ�� word���� �ߺ��� ������� �ʰ� �� �ܾ�� ������ id�� �ο��� dict
print(id_to_word) #{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

def creat_contexts_target(corpus,window_size=1):
    target = corpus[window_size:-window_size] # window�� �ε����� �� Ÿ���� ������! # Ÿ��(���� ���̺�) ����Ʈ �߰� #������ ����� �̿��� �����ְ� �߰��� �ڵ�
    contexts = [] # ���� ���� ����Ʈ

    for idx in range(window_size,len(corpus)-window_size): # idx == target�� index
        cs = []
        for t in range(-window_size,window_size+1): # t�� window�� index
            if t == 0:
                continue
            cs.append(corpus[idx+t]) # window�� -window_size~window_size���� �̵��ϸ� target(idx)�� t(window index)�� ���� ���� cs����Ʈ �߰�(�ƶ� �߰�)
        contexts.append(cs)

    return np.array(contexts), np.array(target)

context,target = creat_contexts_target(corpus,window_size=1)
print(context) # ���� �Է� �����͵�
# [[0 2]
#  [1 3]
#  [2 4]
#  [3 1]
#  [4 5]
#  [1 6]]
print(target) # [1 2 3 4 1 5] #���� ���� ���̺��
# ������ �ܾ� ID... => ��ǻ�Ͱ� ������ �� �ְ� ���ֺ���ȭ!


### ���� ǥ������ ��ȯ
# �ƶ��� Ÿ���� �����ͼ�Ʈ�� '���� ����'�� ��ȯ�ؾ� ��ǻ�Ͱ� ������ �� �ִ�.
# �ƶ��� ���� ���ͷ� ��ȯ�� �� ������ ���� ������ ��ȯ�ȴٴ� ���̴�.
# ex) "You say goodbye and I say hello."
# �ƶ�(6,2) Ÿ��(6,)�� �������� �Ѵٸ� => �ƶ�(6,2,7) ���� (6,7) (2x7 ����� 6��) 
# �����翡���� conver_one_hot() �Լ��� ��� �Ķ���ͷ� '�ܾ�ID���'�� '���� ��'�� ����

import sys
sys.path.append("C:\\Users\\rlaxo\\Desktop\\deepscratch")
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus,window_size=1)

print(contexts)
print(target)

vocab_size = len(word_to_id)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)

print('-------after one-hot--------')

print(contexts)
print()
print(target)


# [[0 2]
#  [1 3]
#  [2 4]
#  [3 1]
#  [4 5]
#  [1 6]]
# [1 2 3 4 1 5]
# -------after one-hot--------
# [[[1 0 0 0 0 0 0]
#   [0 0 1 0 0 0 0]]

#  [[0 1 0 0 0 0 0]
#   [0 0 0 1 0 0 0]]

#  [[0 0 1 0 0 0 0]
#   [0 0 0 0 1 0 0]]

#  [[0 0 0 1 0 0 0]
#   [0 1 0 0 0 0 0]]

#  [[0 0 0 0 1 0 0]
#   [0 0 0 0 0 1 0]]

#  [[0 1 0 0 0 0 0]
#   [0 0 0 0 0 0 1]]]

# [[0 1 0 0 0 0 0]
#  [0 0 1 0 0 0 0]
#  [0 0 0 1 0 0 0]
#  [0 0 0 0 1 0 0]
#  [0 1 0 0 0 0 0]
#  [0 0 0 0 0 1 0]]



