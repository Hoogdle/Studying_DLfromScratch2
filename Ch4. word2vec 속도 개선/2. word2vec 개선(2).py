### word2vec 개선(2) ###
# 은닉층 이후의 처리(행렬곱과 Softmax)
# 네거티브 샘플링 기법을 사용하여 해결!
# Softmax 대신 네거티브 샘플링을 사용하면 어휘가 아무리 많아도 계산량을 낮은 수준에서 일정하게 억제할 수 있다.


### 다중 분류에서 이진 분류로
# 다중 분류를 이진 분류로 근사하는 것이 네거티브 샘플링의 key point!
# before
# Q. 맥락이 you와 goodbye 일 때 타깃 단어는 무엇입니까?
# A. 확률상 say입니다!
# Now
# Q. 맥락이 you와 goodbye 일 때 타깃 단어가 say 입니까?
# A. Yes or No
# 단어가 100만개 있고 W(in)이 100만 x 100이라면 W(out)은 자동으로 100x100만 이다.
# 하지만 이진 분류를 위해서 W(out)을 100x1로 한다! (say에 해당하는 열만 추출!)
# 이후 sigmoid 를 거쳐 확률을 얻는다!

# 입력층 100만 은닉층 100 출력층 100만 일 때
# Embedding의 효과로 은닉층의 벡터 사이즈는 1x100인 벡터(단어 say라 하자)가 된다.
# W(out) (100 x 100만) 에서 say에 해당하는 행만을 선택한다 => 100x1 벡터
# 위의 두 벡터를 곱하면 say의 해당하는 값만을 산출할 수 있다!


### 시그모이드 함수와 교차 엔트로피 오차
# 이진 분류 문제를 풀려면 점수를 시그모이드 함수로 확률로 만들고 손실함수로써 '교차 엔트로피 오차'를 사용한다.(가장 흔하게 사용됨)
# 여기서의 loss는 다음과 같다.
# L = -(tlogy + (1-t)log(1-y)) (t는 정답 레이블, y는 시그모이드 함수의 출력)

# sigmoid와 cross entrophy를 거치는 cal graph에서 Loss를 L이라 했을 때  ∂L/∂x = y-t이다.
# y는 시그모이드의 출력값(확률) t는 정답 레이블로 두 차이가 커지면 가중치를 더 많이 조절하고 (크게 학습) 차이가 작아지면 가중치를 더 적게 조절한다(작게 학습)



### 다중 분류에서 이진 분류로(구현)
# 은닉층과 정답에 해당하는 W(out) index의 Embedding Dot
class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W) #Embedding 클래스 사용
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forwrad(self,h,idx):
        target_W = self.embed.forward(idx) # 미니배치를 고려, idx가 리스트 형태일 수도 있음
        out = np.sum(target_W * h,axis=1) # 넘파이의 * 연산은 원소별 곱을 수행한다.

        self.cache = (h,target_W) # backward에서 사용할 수 있게 h와 target_W 저장하기
        return out # target_W가 다중 numpy 배열이라면 다중 numpy 배열이 리턴됨
    
    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h # 넘파이의 * 연산은 원소별 곱을 수행한다.
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    

### 네거티브 셈플링
# 앞의 과정을 통해 다음과 같은 모델을 만들었다.
# Q. 맥락이 you와 gddobye 일 때 타깃 단어가 say 입니까?
# A. Yes or No
# 이 모델은 you와 goodbye가 주어졌을 때 say의 확률을 높이는 모델은 맞지만 you와 goodbye가 주어졌을 때 다른 단어(ex)hello)에게는 어떠한 패널티도 주지 못합니다.
# 즉 이제 필요한 것은 타깃 단어가 아닌 단어가 나올 확률을 줄이는 tool이 필요합니다.

# cf) 다중 분류문제를 이진 문제로 다룰 수 있으려면 정답과 오답 각각에 대해 바르게 분류할 수 있어야 한다. 지금까지 한 작업은 정답만을 분류한 작업을 했다.
# 긍정의 처리는 타깃 단어에 대해서만 (단 1개) 처리하면 됬다. 하지만 부정의 처리에서는 타깃 단어를 제외한 모든 단어를 처리해야 한다. => 개산량 too big...
# 따라서 모든 단어가 아닌 몇 개의 부정적 대상을 샘플링하여 사용한다. == 네거티브 샘플링
# 네거티브 샘플링 기법은 긍정(타깃)의 경우 손실을 구하고 동시에 부정적 예를 몇 개 샘플링 하여 부정적인 예에 대해서도 손실을 구한다.
# 그 후 각각의 데이터의 손실을 더한 값을 최종 손실로 한다.
    
# ex) You say goodbye and I say hello. 타깃 : say, 샘플링 : hello, I
# say를 Embedding Dot 할 때는 정답 레이블을 1로 
# hello와 I로 Embedding Dot 할 때는 정답 레이블을 0으로
# 모두의 경우의 손실을 다 더 하여 최종손실을 산출한다.
    

### 네거티브 샘플리의 샘플링 기법
# 그렇다면 어떤 단어들을 샘플링의 후보로 둘것인가? 
# 말뭉치에서 단어의 출현 빈도수를 기준으로 각 단어의 출현 확률 분포를 구한 후 확률 분포대로 네거티브 단어를 샘플링한다.
# => 자주 나오는 단어가 선택될 가능성이 높기 때문에 희소한 단어가 선택되기는 어렵다(good)

# 확률 분포에 따라 샘플링하는 코드 구현
import numpy as np

# 0~9 까지의 숫자 중 하나를 무작위로 샘플링
print(np.random.choice(10)) #7

# 단어 리스트에서 무작위로 단어 하나 샘플링
words = ['you','say','goodbye','I','hello','.']
print(np.random.choice(words)) #hello

# 5개만 무작위로 샘플링(중복존재) size로 크기 조절
print(np.random.choice(words,size=5)) #['hello' 'I' 'you' 'say' 'you']

# 5개만 무작위로 샘플링(중복없음) size로 크기 조절
print(np.random.choice(words,size=5,replace=False)) #['hello' 'say' 'goodbye' 'you' 'I']

# 확률 분포에 따라 샘플링
p = [0.5,0.1,0.05,0.2,0.05,0.1]
print(np.random.choice(words,size=3,p=p)) #['you' 'you' 'you']

print(np.random.choice(words,size=3,replace=False)) #['you' 'say' '.']


# 이때 분포가 작은 값이 샘플리에 참여하게 하기 위해 각 단어의 분포에 0.75제곱을 해준다. => 분포가 낮은 단어의 분포가 올라간다!
# P'(w.i) = (P(w.i)^0.75) / ((j~n).Σ (P(w.i)^0.75))

p = [0.7,0.29,0.01]
new_p = np.power(p,0.75)
new_p /= np.sum(new_p)
print(new_p) #[0.64196878 0.33150408 0.02652714]

# 해당 교재에서는 UnigramSampler 클래스로 위 코드들을 구현
# UnigramSampler 클래스의 매개변수는 단어 ID 목록인 'Corpus', 확률분포에 제곱을 할 값인 'power', 샘플링 할 데이터의 갯수인 'sample_size'가 있다.\
# get_negative_sample 이라는 메서드를 통해 target 인수의 단어들을 긍정적인 예로 인식하고 그 외의 단어 ID를 샘플링한다.
# 사용예시
corpus = np.array([0,1,2,3,4,1,2,3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus,power,sample_size)
target = np.array([1,3,0]) # 타깃들을 미니배치로!
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)
# [[0 3] # 1이 타깃일 때의 sample
#  [1 2] # 3이 타깃일 때의 sample
#  [2 3]] # 0이 타깃일 때의 sample



### 네거티브 샘플링 구현
# NegativeSamplingLoss 클래스로 구현
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

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h,target)
        corret_label = np.ones(batch_size,dtype=np.int32)
        loss = self.loss_layers[0].forward(score,corret_label)

        # 부정적 예 순전파
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

### CBOW 모델 구현
    
