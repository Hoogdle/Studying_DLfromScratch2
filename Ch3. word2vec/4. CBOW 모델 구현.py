### CBOW 모델 구현 ###


import sys
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')
import numpy as np
from common.layers import Matmul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H = vocab_size,hidden_size

        #가중치 초기화
        W_in = 0.01 * np.random.rand(V,H).astype('f')
        W_out = 0.01 * np.random.rand(H,V).astype('f')

        #계층 생성
        self.in_layer0 = Matmul(W_in)
        self.in_layer1 = Matmul(W_in)
        self.out_layer = Matmul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모음
        layers = [self.in_layer0,self.in_layer1,self.out_layer,self.loss_layer]
        self.params, self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 인스턴스 변수에 단언의 분산 표현을 저장
        self.word_vec = W_in

    def forward(self,contexts,target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0+h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss
    
    def backward(self, dout=1): #backward를 실행하는 것만으로도 gards 리스트의 기울기가 갱신
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None

### 학습 코드 구현

import sys
import numpy as np
sys.path.append('C:\\Users\\rlaxo\\Desktop\\deepscratch')

from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H = vocab_size,hidden_size

        #가중치 초기화
        W_in = 0.01 * np.random.rand(V,H).astype('f')
        W_out = 0.01 * np.random.rand(H,V).astype('f')

        #계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모음
        layers = [self.in_layer0,self.in_layer1,self.out_layer,self.loss_layer]
        self.params, self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 인스턴스 변수에 단언의 분산 표현을 저장
        self.word_vec = W_in

    def forward(self,contexts,target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0+h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss
    
    def backward(self, dout=1): #backward를 실행하는 것만으로도 gards 리스트의 기울기가 갱신
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None

from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess,create_contexts_target,convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts,target = create_contexts_target(corpus,window_size)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)

model = SimpleCBOW(vocab_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()

# 단어 벡터 표시하기
word_vec = model.word_vec
for word_id, word in id_to_word.items():
    print(word,word_vec[word_id])

# you [1.0454618 1.0132304 1.0875694 1.0460285 1.4164021]
# say [-1.2267221  -1.2232676  -0.76017106 -1.2373134   0.1807015 ]
# goodbye [0.76635057 0.815203   0.9261453  0.77122575 0.5222998 ]
# and [-0.8077458  -0.80605465 -1.4625862  -0.7986305   1.7670093 ]
# i [0.77051705 0.81698537 0.9301936  0.765249   0.5179494 ]
# hello [1.0442084 1.0082895 1.0891881 1.0439935 1.4229944]
# . [-1.2659308 -1.2610803  1.4324841 -1.280068  -1.5577202]
    
# 위 벡터야 말로 단어의 밀집벡터이고 단어의 분산 표현이다.
# 지금은 단일 문장으로 학습했기에 정확도가 낮지만 많은 데이터를 넣을 수록 정확도는 올라간다.
# 하지만 CBOW는 처리 효율에서 문제가 있다..!