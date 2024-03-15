### 계산 고속화 ###

### 비트 정밀도
# 넘파이의 부동소수점 수는 기본적으로 64비트를 사용한다
import numpy as np
a = np.random.rand(3)
print(a.dtype) #float64 # == 64비트 부동소수점 수

# 하지만 메모리관점(계산속도,버스방목)에서는 32비트가 더 효율적이다.
# 따라서 이 교재에서는 32비트 부동소수점 수를 우선으로 사용한다
# 32비트 부동소수점을 사용하려면 np.float32나 f로 지정
b = np.random.randn(3).astype(np.float32)
print(b.dtype) #float32
c = np.random.randn(3).astype('f')
print(c.dtype) #float32

# 16비트가 더 효율적이긴 하지만 CPU와 GPU연산은 대게 32비트에서 진행하기에 가중치 저장시에만 16비트 사용



### GPU 쿠파이
# 딥러닝의 계산은 대량의 곱하기 연산으로 구성됨. 이는 병렬로 처리해야 효율적 => GPU가 CPU 보다 효율적!
# '쿠파이' 라이브러리를 통해 GPU 병렬 계산 가능
# 쿠파이와 넘파이는 호환되는 API 제공!(100%는 아님)
import cupy as cp
# 쿠파이는 기본적으로 넘파이와 사용법이 같다