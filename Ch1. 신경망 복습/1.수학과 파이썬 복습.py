### 벡터와 행렬
# 벡터, 파이썬에서는 1차원 배열로 취급
# 행렬, 2차원 배열로 취급
# 벡터는 열(세로) 행(가로)로 표현 가능하다(열벡터,행벡터) (구현의 편의를 위해 이 책에서는 '행벡터'로 다룸)
# 파이썬에서 벡터를 '행벡터'로 취급하는 경우 벡터를 가로 방향 '행렬'로 변환해 사용하면 명확해진다. (N요소 벡터 => 1XN 형상의 행렬)

# import numpy as np

# x = np.array([1,2,3]) 
# print(x.__class__) #<class 'numpy.ndarray'> #np.ndarray 클래스에는 다양한 메서드와 인스턴스 변수 존재
# print(x.shape) #(3,) #np.ndarray의 인스턴스 변수 shape
# print(x.ndim) #1 #np.ndarray의 인스턴스 변수 ndim

# W = np.array([[1,2,3],[4,5,6]])
# print(W.shape) #(2, 3)
# print(W.ndim) #2


### 행렬의 원소별 연산
# import numpy as np

# W = np.array([[1,2,3],[4,5,6]])
# X = np.array([[0,1,2],[3,4,5]])
# print(W+X) # 대응하는 원소끼리(독립적으로) 연산
# [[ 1  3  5]
#  [ 7  9 11]]
# print(W*X) # 대응하는 원소끼리(독립적으로) 연산
# [[ 0  2  6]
#  [12 20 30]]


### 브로드캐스트
# 형상이 다른 배열끼리도 연산 가능
# import numpy as np

# A = np.array([[1,2],[3,4]])
# print(A*10) # 스칼라 10이 A의 형상에 맞게 변경((2,2)행렬로 변환)
# [[10 20]
#  [30 40]]
# A = np.array([[1,2],[3,4]])
# b = np.array([10,20])
# print(A*b) # 행렬 곱 연산 X, b가 브로드캐스팅 되어 (2,2) 행렬로 변환 이후 각 원소끼리 연산
# b가 아래와 같이 변함
# [[10 20]
#  [10 20]]
# [[10 40]
#  [30 80]]


### 벡터의 내적과 행렬의 곱
# 벡터의 내적 == 두 벡터의 대응 원소끼리의 곱을 더한 것
# np.dot() 과 np.matmul() 메서드로 구현
# import numpy as np
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(np.dot(a,b)) #32
# A = np.array([[1,2],[3,4]])
# B = np.array([[5,6],[7,8]])
# print(np.matmul(A,B))
# [[19 22]
#  [43 50]]

# 행렬곱도 np.dot()로 구현 가능
# np.dot() 의 인수가 1차원, 벡터의 내적
# np.dot() 의 인수가 2차원, 행렬ㅇ릐 곱


