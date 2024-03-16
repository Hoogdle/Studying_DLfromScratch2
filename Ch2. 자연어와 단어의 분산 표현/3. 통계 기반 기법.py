### ��� ��� ��� ### 

# ����ġ(corpus) : �뷮�� �ؽ�Ʈ ������
# ����ġ ���� �����͵��� ����� �� ���̱⿡ �ڿ�� ���� ����� '����'�� ����� ����ִٰ� �� �� �ִ�.
# ex) ������ ���� ���, �ܾ �����ϴ� ���, �ܾ��� �ǹ�
# ��� ��� ����� ����ġ���� �ڵ�����, ȿ�������� �ٽ��� �����ϴ� ���̴�.


### ���̽��� ���V�� ��ó���ϱ�
# ��ó�� : �ؽ�Ʈ �����͸� �ܾ�� �����ϰ� ���ҵ� �ܾ���� �ܾ� ID ������� ��ȯ�ϴ� ��

text = 'You say goodbye and i say hello.' # ����ġ # �������� �̷��� ����ġ�� ��õ~������
text = text.lower()
text = text.replace('.',' .')
print(text) #you say goodbye and i say hello .
words = text.split()
print(words) #['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']

# �ܾ� ������ ���ҵǾ� �ٷ�� ����������, �ؽ�Ʈ �״�� �����ϱ⿡�� ���� => �ܾ�� ID �ο�
# ID�� ����Ʈ�� �̿��� �� �ֵ��� ����!

word_to_id = {} #�ܾ�� ID��ȯ ���
id_to_word = {} #ID���� �ܾȯ ���

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(id_to_word) #{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
print(word_to_id) #{'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}

# id�� Ȱ���� �ܾ �˻��ϰų�, �ܾ ���� id�� �˻�!
print(id_to_word[1]) #say
print(word_to_id['say']) #1

# �ܾ��� => ID ���
import numpy as np
corpus = [word_to_id[w] for w in words] 
print(corpus) #[0, 1, 2, 3, 4, 1, 5, 6] #����Ʈ
corpus = np.array(corpus)
print(corpus) #[0 1 2 3 4 1 5 6] ������ �迭

# ���� ���μ����� �ϳ��� �Լ���(preprocess)
def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split() # split�� ����Ʈ���� ����
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])

    return corpus,word_to_id,id_to_word

# ��뿹��
text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
# corpus == ID ���
# word_to_id == �ܾ� : ID
# id_to_word == ID : �ܾ�
# ��ó�� ���� �Ϸ�!(����ġ�� �ٷ� �غ� �Ϸ�!) => ����ġ�� ����� �ܾ��� �ǹ̸� �����ؾ���!



### �ܾ��� �л� ǥ��
# ��� ���� ������ �̸��� �ִ�(�ڹ�Ʈ ���, ��ũ����) 
# �̸���� ���� ǥ���ϴ� ���, RGB(����)�� ����Ͽ� ���� ǥ���ϴ� ����� �����Ѵ�.
# RGB�� ǥ�� �ϴ� ����� �� �� �������̸� ��Ȯ�ϰ� ����� �� �ִ�.(��� == RGB(170,33,22), ����� ���� �� ���� ���� �������� �� �� ����)
# �׷� �ܾ ���ͷ� ǥ���� �� ���� ������?
# �л�ǥ��(distrbut;ional representation) : �ܾ��� �ǹ̸� ��Ȯ�а� �ľ��� �� �ִ� ���� ǥ��
# �л�ǥ���� ���� ������ �������͸� ����Ѵ�.([0.21, -0.45, 0.83]), �ܾ��� �л�ǥ���� ��� ������ �������� ����Ʈ


### ���� ����
# �ڿ��� ó���� �ֿ� ����� ��� �� �ϳ��� ������ ���̵� �Ѹ��� �ΰ� �ִ�. => ��������
# �������� : �ܾ��� �ǹ̴� �ֺ� �ܾ ���� �����ȴ�.
# ��, �ܾ� ��ü�δ� �ǹ̰� ���� �� �ܾ ���� �ƶ��� �ǹ̸� �����Ѵ�.
# ex) I drink beer, We drin wine ==> drink �ֺ����� ���ᰡ �����ϱ� ����.
# ex) I guzzle beer, We guzzle wine ==> guzzle�� drink�� ���� �ƶ����� ���Ǳ���!(guzzle�� drink�� ����� �ǹ� �ܾ��̱���!)
# �ƶ� : Ư�� �ܾ �߽��� �� �� �ֺ��ܾ�(������ ũ�⿡ ���� �޶���)
# ex) You say goodbye and i say hello. ==> goodbye�� �߽��� �� window size = 1, say and�� �ƶ��� ����
# ���⼭�� �¿�� �Ȱ��� ���� �ܾ �ƶ����� ��������� ��Ȳ�� ���� ���� �ܾ Ȥ�� ������ �ܾ ����ϱ⵵ �ϸ� ������ ���۰� ���� ����ϱ⵵ �Ѵ�.
# �� å������ ���� ���ظ� ���� �¿� ������ �ƶ����� ���


### ���ù߻� ���
# �������� �ܾ ���ͷ� ��Ÿ���� ����� �����غ���.

# ��� ��� ���(�ָ� �ܾ� ������ � �ܾ �� ���̳� �����ߴ°�)
import sys
sys.path.append('...')
import numpy as np
from common.util import preporcess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preporcess(text)

print(corpus) #[0 1 2 3 4 5 6]
print(id_to_word)
# {0 : 'you', 1: 'say', 2:'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

# ������ ũ�Ⱑ 1�̰� ID�� 0 �� you���� ����ݱ���� ����Ѵٸ�

#       you     say     goodbye     and     i       hello       .
# you   0       1       0           0       0       0           0
# you��� �ܾ [0,1,0,0,0,0,0] ��� ���ͷ� ǥ���� �� �ִ�.

# �ش� �۾��� ��� �ܾ� �����Ѵٸ� =>
#           you     say     goodbye     and     i       hello       .
# you       0       1       0           0       0       0           0
# say       1       0       1           0       1       1           0
# goodbye   0       1       0           1       0       0           0
# and       0       0       1           0       1       0           0
# i         0       1       0           1       0       0           0
# hello     0       1       0           0       0       0           1
# .         0       0       0           0       0       1           0

# �� ǥ�� �� ���� �ش� �ܾ ǥ���ϴ� ���Ͱ� �ȴ�.
# ���� ǥ�� '���ù߻� ���(co-occurrence matrix)'��� �Ѵ�.

# ���ù߻���� �̿�
# C�� ���ù߻������ ��(����� ����)
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
],dtype=np.int32)

# ���ù߻� ����� ����ϸ� �ܾ��� ���͸� ���� ���� �� �ִ�.
print(C[0]) #ID�� 0�� �ܾ��� ���� ǥ��
#[0 1 0 0 0 0 0]
print(C[4]) #ID�� 4�� �ܾ��� ���� ǥ��
#[0 1 0 1 0 0 0]
print(C[word_to_id['goodbye']]) #"goodbye"�� ���� ǥ��
#[0 1 0 1 0 0 0]

# ����ġ�κ��� ���ù߻� ����� ������ִ� �Լ� ����
def creat_co_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id][left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id][right_word_id] += 1

    return co_matrix
        

# ���� �� ���絵
# �տ����� ���ù߻� ����� Ȱ���� �ܾ ���ͷ� ǥ���ϴ� ����� �˾ƺõ� �������� ���� ������ ���絵�� �����ϴ� ����� �˾ƺ� ���̴�!
# ������ ���絹�� ��Ÿ�� ���� '�ڻ��� ���絵'�� ���� �̿��Ѵ�.
# x dot y = ||x|| ||y|| cos(theta) ����
# simlarity(x,y) = (x dot y) / (||x|| ||y||) # �� ���� �ٽ��� ���͸� ����ȭ�ϰ� ������ ���ϴ� ��.

# ����
# x,y�� ������ �迭
def cos_similarity(x,y):
    nx = x / np.sqrt(np.sum(x**2)) 
    ny = y / np.sqrt(np.sum(y**2))
    return np.dot(nx,ny)
# ���� �ڵ�� x�Ǵ�y�� 0�����̸� �и� 0�� �Ǿ� ������ ����� 
# eps(=0.00000001)�� �и� �����༭ �ذ�!
def cos_similarity(x,y):
    nx = x / (np.sqrt(np.sum(x**2))+eps)
    ny = y / (np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx,ny)

# ��������
# you�� i�� ���絵�� ���ϴ� �ڵ�
import sys
sys.path.append('...')
from common.util import preprocess, create_co_matrix, cos_similarity
text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preporcess(text)
vocab_size = len(word_to_id)
C = creat_co_matrix(corpus,vocab_size)
c0 = C[id_to_word['you']] # ����ݱ���� ��ģ you�� ����
c1 = C[id_to_word['I']] # ����ݱ���� ��ģ I�� ����
print(cos_similarity(c0,c1)) # 0.7071067691154799 #cos_simlar�� -1~1 ������ ���� �����Ƿ� ���� ���絵�� ���ٰ� �� �� �ִ�.


### ���� �ܾ��� ��ŷ ǥ��
# �ܾ �־����� �� �ش� �ܾ�� ����� �ܾ ���絵 ������ ����ϴ� �Լ� ����

# �Ķ���� ����
# query �˻���(�ܾ�), word_to_id, id_to_word(), word_matrix(�ܾ� ���͵��� ���� ���), top(���� �� ������ ������� ����)

def most_similar(query,word_to_id,id_to_word,word_matrix,top=5):
    if query not in word_to_id:
        print(f'{query}�� ã�� �� �����ϴ�.')
        return
    
    print(f'\n[query]'+query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)

    count = 0
    for i in (-1*similarity).argsort():
    # argsort() �޼��� == ������������ ����
    # �̶� sdimilarity�� -1�� ���ϰ� ��������(ũ�⸦ ������ �ϰ� ��������) => �������� ����
    # argsort()�� ��ȯ���� ũ�⺰ ���ĵ� ������ index ����Ʈ�̴�.
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word} : {similarity[i]}')

        count += 1
        if count>=top:
            return


### ��������
# import sys
# sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch\\')
# from common.util import preprocess, create_co_matrix, most_similar

# text = "You say goodbye and I say hello."
# corpus,word_to_id,id_to_word = preprocess(text)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)

# print(most_similar('you',word_to_id,id_to_word,C,top=5))

# [query] you
#  goodbye: 0.7071067691154799
#  i: 0.7071067691154799
#  hello: 0.7071067691154799
#  say: 0.0
#  and: 0.0
        
# you�� i�� ��Ī ����� ���� ����Ѱ��� ���ذ� ������ you�� goodbye,hello�� ���絵�� ���� ���� �̻��ϴ�.
# ���� ������ ����ġ�� ũ�Ⱑ �ʹ� �۴ٴ� ���� �����̱� �ϴ�.



