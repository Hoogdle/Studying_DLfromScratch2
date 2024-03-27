### 언어 모델을 사용한 문장 생성 ###
# 언어 모델로 문장을 생성하기!

### RNN을 사용한 문장 생성의 순서
# ex) You say goodbye and I say hello. 라는 문장으로 학습된 RNN 모델이 있다고 가정하자
# 이 모델에게 'I'라는 입력 값을 주면 각 단어의 확률 분포에서 say의 확률이 가장 높을 것이다.
# 확률이 가장 높은 단어를 선택하는 '결정적인 방법' 보다는 확률인 높은 단어일수록 선택될 확률이 높고, 낮은 단어일수록 선택될 확률이 낮은 '확률적 선택' 방법을 사용한다(이 책에서)
# I의 출력값으로 say가 나올 것이고 say의 출력값으로 hello 가 나올 것이다.(나올 확률이 높다.)
# 여기서 생성된 문장은 훈련 데이터에는 존재하지 않는, 새로 생성된 문장이다.(언어 모델은 훈련 데이터를 암기하는 것이 아닌 사용된 단어의 정렬 패턴을 학습하기 때문)


### 문장 생성 구현

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
            x = np.array(x).reshape(1,1) # RNN의 input은 대게 2차원의 형태, 따라서 x를 2차원으로 변경해줌
            score = self.predict(x)
            p = softmax(score.flatten())
            
            sampled = np.random.choice(len(p),size=1,p=p) #len(p)를 통해 word의 id가 선택됨. p는 확률 리스트
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids