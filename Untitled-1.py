# %% 
a = input("첫번째 숫자 입력하기: ") ## a 숫자 입력
b = input("두번째 숫자 입력하기: ") ## b 숫자 입력
print("입력한 숫자는 다음과 같습니다: ", a , " , " , b) ## a 와 b에 입력된 숫자 출력
print("입력한 숫자의 자료형은 다음과 같습니다: ", type(a), type(b)) ## a와 b의 자료형(문자열)
print("입력한 숫자의 사칙연산을 진행합니다.")
print(a ," + ", b, " = ", float(a) + float(b), end = " | ") ## float()를 이용해 자료형을 문자열에서 숫자(실수)로 변경
print(a ," - ", b, " = ", float(a) - float(b), end = " | ") ## 소숫점을 표시하기 위해 float 사용함
print(a ," * ", b, " = ", float(a) * float(b), end = " | ") ## end 를 넣으면 줄바꿈 대신 끝에 해당 기호를 붙인다
print(a ," / ", b, " = ", float(a) / float(b), end = " | ")
# %%
for i in range(11): ## i를 0부터 10까지 반복한다
    if i < 10: ## i가 10보다 작을 때
        print(10 - i, end=",") ## 10에서 i를 빼고 뒤에 콤마를 붙인다
    else: print(10 - i) ## i가 10 이상일 때
for j in range(9):
    print((j+1)*10) ## j는 0부터 시작하기 때문에 10부터 출력하기 위해 j+1을 해줬다
# %%
a = [68, 97, 116, 97, 32, 83, 116, 114, 117, 99, 116, 117, 114, 101]
for i in range(len(a)): ## len을 이용해 a의 길이만큼 반복
    b = chr(a[i]) ## a의 i번째 수를 아스키코드 문자로 변환
    print(b, end= "") ## 변환된 아스키코드 문자 출력
# %%
import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.add(a,b) #c=a+b
print(c)
d = np.dot(a,b)
print(d)
# %%
import numpy as np
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

print(arr1)
print(arr1+arr2)
print(arr1+5)
print(arr1-arr2)
print(arr1-5)
print(arr1*arr2)
print(arr1*2)
print(arr1/arr2)
print(arr1/5)

# %%
import numpy as np
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]

print(list1*2)

# %%
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7])

print(a.sum()) # 합 
print(a.mean()) # 평균
print(a.var()) # 분산
print(a.std()) # 표준편차
print(a.min())
print(a.max())
print(a.cumsum())
print(a.cumprod())

# %%

import numpy as np
a = np.tile(5, 9)
print(a)
a = np.full(5, 9)
print(a)
a = np.eye(3, dtype=int)
print(a)


# %%
def seq(n):
    if n == 1:
        return 1
    else :
        return seq(n-1) + 3
    
a = seq(int(input("enter: ")))
print("number is : ", a)

# %%
# %%writefile bubbleSort.py
def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):

            # Traverse the array from 0 to n-i-1. Swap if the element found
            # is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
# %%
import time
import bubbleSort

# 임의의 값 10개 생성
arr = [9, 38, 3, 1, 189, 725, 72, 3, 37, 10] ## 임의의 수 10개

start = time.time() ## 시작하는 시간을 start에 저장
bubbleSort.bubbleSort(arr) ## bubbleSort 파일의 bubbleSort 함수 실행
end = time.time() - start ## 끝나는 시간에서 시작하는 시간을 빼면 실행 시간이 나온다

print("정렬: ", arr) ## 정렬 된 10개의 수를 출력
print("시간 측정: ", end, " 초") ## 실행 시간 출력

# %%
def fib(n):
    fib[1] <- fib[2] <- 1
    for i in range(3, n):
        fib[i] <- fib[i-1] <- fib[i-2]
        return fib[n]
a = fib(int(input()))
print(a)
# ?????
# %%
# def hanoi(n, a, b, c):



# %%

import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a * 2)

a = np.array([1, 2, 'A'])
print(a)

a = np.array([1, 2, 3.14])
print(a)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)

a = np.array([1, 2, 3, 4, 5])
print(a.size)

# %%

a = np.arange(start=1, stop=9, step=2) #시작은 포함하고 끝은 포함되지 않는다
# 1 <= a < 9
print(a)

a = np.arange(0, 0.8, 0.2) # 0 <= a < 0.8
print(a)

a = np.arange(3)
print(a)

# %%

import numpy as np

a = np.tile('A', 3)
print(a)

a = np.tile(7, 3)
print(a)

a = np.full(7, 3)
print(a)

a = np.zeros(3)
print(a)

a = np.zeros([2, 3])
print(a)

a = np.ones([2, 3])
print(a)

# %%

import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[0])

print(a[1:3]) # a의 인덱스 1이상 3미만 1 <= a < 3

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(a[0, 3])

print(a[1, 2:4])
print(a[0, 2])

# %%

import numpy as np
import pandas as pd

df = pd.DataFrame(
    [[1,2],
     [3,4],
     [5,6]],
    columns=['a','b']
    )
print(df)

arr = np.array([1, 2, 3, 4, 5])

df = pd.DataFrame({
    'col1' : arr,
    'col2' : arr*2,
    'col3' : ['A', 'B', 'C', 'D', 'E']
    })
print(df)

# %%

import numpy

a = numpy.array=([1,2,3])
print(a)

# %%

import pandas as pd

file_df = pd.read_csv("C:/Users/김보석/Desktop/sample/sample1.csv")
file_df2 = pd.read_csv("C:/Users/김보석/Desktop/sample/sample2.txt", encoding="UTF-8")
file_df3 = pd.read_csv("C:/Users/김보석/Desktop/sample/sample3.txt", sep = '\t', encoding="UTF-8")
file_df4 = pd.read_excel("C:/Users/김보석/Desktop/sample/sample4.xlsx")
file_df5 = pd.read_csv("C:\\Users\\김보석\\Desktop\\sample\\sample1.csv")


print(file_df4)


# %%

import numpy as np
import pandas as pd

df1 = pd.DataFrame({
    'col1' : np.array([1, 2, 3]),
    'col2' : np.array(['A', 'B', 'C'])
})

df2 = pd.DataFrame({
    'col1' : np.array([4, 5, 6]),
    'col2' : np.array(['D', 'E', 'F'])
})

print(pd.concat([df1, df2], '\n'))




# %%

global n # n을 전역변수로 선언

def recursion(s, l, r):
    global n # 전역변수를 사용한다고 선언
    n += 1 # 해당 코드를 불러올 때 마다 n에 1씩 더한다
    if l >= r: return 1
    elif s[l] != s[r]: return 0
    else: return recursion(s, l+1, r-1)

def isPalindrome(s):
    return recursion(s, 0, len(s)-1)
count = int(input())

inin = [] # 리스트 생성
for i in range(0, count):
    check = input() # check에 문자를 받는다
    inin.append(check) # inin에 받은 문자를 뒤로 집어넣는다
for i in range(0, count):
    n = 0 # 반복될 때 마다 0으로 초기화
    print(isPalindrome(inin[i]), n) # inin[i]번째 문자를 s값에 넣고 함수를 호출한다 recursion 함수가 돌아간 횟수만큼 n이 나온다




# %%

import pandas as pd
df = pd.DataFrame({
'col1' : [1,2,3,4,5],
'col2' : [2,4,6,8,10],
'col3' : ['A','B','C','D','E'] })
print(type(df.col2.values))





# %%

import numpy as np

a = np.zeros(5, dtype=int)
print(a)

a = np.zeros((3, 3), dtype=int)
print(a)

a = np.ones(5, dtype=float)
print(a)

# %%

import numpy as np

a = np.tile(5, 9)
print(a)

a = np.full(5, 9)
print(a)

a = np.eye(3, dtype=int)
print(a)


# %%

import numpy as np

a = np.arange(20)
print(a)

print(a.reshape(4,5))

print(a)


# %%

import numpy as np

a = [1, 'A']

print(type(a[0]))
print(type(a[1]))

a = np.array(a)

print(type(a[0]))
print(type(a[1]))


# %%

import pandas as pd


df1 = pd.DataFrame({
    'a' : [1, 2, 3],
    'b' : [4, 5, 6]
})
df2 = pd.DataFrame({
    'a' : [7, 8, 9],
    'b' : [10, 11, 12]
})

cc = pd.concat([df1, df2], axis=0)
print(cc)

cc = pd.concat([df1, df2], axis=1)
print(cc)

df3 = pd.DataFrame({
    'c' : [7, 8, 9],
    'd' : [10, 11, 12]
})

cc = pd.concat([df1, df3], axis=0)
print(cc)

cc = pd.concat([df1, df3], axis=1)
print(cc)


# %%

import pandas as pd

a = pd.DataFrame(
    [['A', 1], ['B', 2], ['C', 3]],
    columns = ['x1', 'x2']
)
b = pd.DataFrame({
    'x1' : ['B', 'C', 'D'],
    'x3' : [2, 3, 4]
})

print(a,'\n')
print(b,'\n')
print(pd.concat([a,b]), '\n')
print(pd.concat([a,b], axis=1), '\n')


# %%

def check(t):
    global count # 전역변수 선언
    if(t == S): # T와 S가 같아지면 
        count = 1 # 1 값 저장
        return # 재귀 탈출
    if(len(t) <= len(S)): # T의 길이가 S 이하로 내려갈 때
        count = 0 # 0 저장
        return # 재귀 탈출
    else: # 여기가 실질적인 재귀함수 작동구역
        #print(t) # 
        if(t[-1] == "A"): check(t[:-1]) # 재귀함수에서는 pop보다 이런 형식이 더 많이 보이는 것 같다 T[:-1]은 맨 뒤 함수를 빼고 리스트를 반환한다
        if(t[0] == "B"): check(t[::-1][:-1]) # 얘도 reverse를 사용하려고 했는데 잘 안되가지고 바꿨다 t[::-1]은 간격을 -1로 설정해서 최종적으로 뒤집는 역할을 한다
                                             # 이후에 [:-1]을 이용해 맨 뒤 함수를 빼고 리스트를 반환한다

S = list(input()) # S 입력
T = list(input()) # T 입력

check(T) # t에 T 입력
print(count) # 값 출력


# %%

# 수 많은 과정의 악수요청


def check(t):
    global count # 전역변수 선언
    if(t == S): # T와 S가 같아지면 
        count = 1 # 1 값 저장
        return # 재귀 탈출
    if(len(t) <= len(S)): # T의 길이가 S 이하로 내려갈 때
        count = 0 # 0 저장
        return # 재귀 탈출
    else: # 여기가 실질적인 재귀함수 작동구역
        #print(t) # 
        if(t[-1] == "A"): check(t[:-1])
        if(t[0] == "B"): check(t[::-1][:-1])

S = list(input())
T = list(input()) 

check(T) # t에 T 입력
print(count) # 값 출력









def first(a):
    a.append("A")
    return a

def second(a):
    a.append("B")
    a = a.reverse()
    return a

S = input()
A = S + 'A'
B = S + "A"
print(A)
print(B)





for i in range(len(T)):
    if(len(S) > len(T)):
        print(0)
        break
    if(S == T): 
        print(1)
        break
    else:
        if(T[len(S)] == "A"): first(S)
        else: second(S)
        
    
    

    





# %%


print("어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.")
output1 = "\"재귀함수가 뭔가요?\""
output2 = "\"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어."
output3 = "마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지."
output4 = "그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어.\""
output5 = "----"
output6 = ""
output7 = "라고 답변하였지."
# output 1 ~ 7 까지 문자 저장
count = 0 # 입력받은 수를 따로 저장받기 위해 새 변수 생성
a = input() # 반복할만큼 입력받음
for i in range(int(a)): # a는 문자로 입력받기 때문에 int를 적어서 숫자로 변경
    print(output6 + output1)
    print(output6 + output2)
    print(output6 + output3)
    print(output6 + output4) # output6은 처음에 공란이기 때문에 이 사이클을 지나서
    output6 += output5 # ---- 를 추가한다
    count += 1 # 사이클이 돌아가는 만큼 count 증가
print(output6 + output1) # 얘랑
print(output6 + "\"재귀함수는 자기 자신을 호출하는 함수라네\"") # 얘는 반복되지 않기 때문에 사이클이랑 분리해놨다
for i in range(int(a)): # 다시 입력받은 수 만큼 반복
    output6 = "" # 여기는 count가 감소하기 때문에 초기화를 한다
    for i in range(count): # count를 점차 감소시키기 위해 새로운 반복문 생성
        output6 += output5 # 위에서 output이 공란이 되기 때문에 여기서 다시 추가한다
    print(output6 + output7)
    count -= 1 # count를 1씩 감소시킨다
print(output7)





# %%

print("어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.")
output1 = "\"재귀함수가 뭔가요?\""
output2 = "\"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어."
output3 = "마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지."
output4 = "그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어.\""
output5 = "____"
output6 = "라고 답변하였지."

def loop1(a): # 재귀함수
    global count # count값 전역함수로 받기 - 얘는 따로 카운트를 저장하기 때문에 global 선언 해야함
    print(output5 * count + output1) # 5번 문자를 count 수 만큼 출력한다 문자열이라 곱셈 가능
    if(count == a): # count 값이 입력받은 숫자와 같아지면 코드 출력
        print(output5 * count + "\"재귀함수는 자기 자신을 호출하는 함수라네\"")
    else: # count값이 입력받은 숫자보다 작을 때 여기 코드 반복
        print(output5 * count + output2)
        print(output5 * count + output3)
        print(output5 * count + output4)
        count += 1 # count 값을 1씩 올린다
        loop1(a) 

def loop2(count): # 재귀함수
    if(count == 0): # count값이 0일 때
        print(output6) # 출력하고 루프 탈출
    else:
        print(output5 * count + output6) # 라고 답변하였지 출력
        count -= 1 # count값 1씩 줄이고
        loop2(count) # 다시 루프

    
count = 0
a = int(input()) # a값 숫자로 입력받는다
loop1(a) # a값 넣는다
loop2(count) # 여기는 count 값을 넣기 때문에 전역함수로 받을 필요 없음





# %%


print("어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.")
output1 = "\"재귀함수가 뭔가요?\""
output2 = "\"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어."
output3 = "마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지."
output4 = "그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어.\""
output5 = "____"
output6 = "라고 답변하였지."

def loop(a, count): # 재귀함수
    print(output5 * count + output1) # 5번 문자를 count 수 만큼 출력한다 문자열이라 곱셈 가능
    if count == a: # count 값이 입력받은 숫자와 같아지면 코드 출력
        print(output5 * count + "\"재귀함수는 자기 자신을 호출하는 함수라네\"")
    else: # count값이 입력받은 숫자보다 작을 때 여기 코드 반복
        print(output5 * count + output2)
        print(output5 * count + output3)
        print(output5 * count + output4)
        loop(a, count + 1) # count 값을 1씩 올린다
    # print(count)
    print(output5 * count + output6) # 

a = int(input()) # a값 숫자로 입력받는다
loop(a, 0) # a값 넣는다


# %%

import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
fish2 = np.array([2,3,3,4,4,4,4,5,5])
print('합:', fish.sum(), np.sum(fish))
print('개수:', fish.size, np.size(fish))
print('평균:',fish.mean(), np.mean(fish) )
print(fish.var())

# %%

import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
print('분산:',fish.var(), np.var(fish) )
print('불편분산:',fish.var(ddof=1), np.var(fish, ddof=1) )
print('표준편차:',fish.std(), np.std(fish) )
print('불편표준편차:',fish.std(ddof=1), np.std(fish, ddof=1) )


# %%

import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
fish1 = [0, 0.25, 0.25, 0, 0, 0, 0, 0.75, 0.75, 1]
fish2 = []

for i in range(fish.__len__()):
    fish2.append((fish[i] - fish.mean())/fish.std())

print(fish2.var())



# %%

import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
fish1 = np.array([0, 0.25, 0.25, 0, 0, 0, 0, 0.75, 0.75, 1])
fish2 = np.array([-1.83, -0.91, -0.91, 0.0, 0.0, 0.0, 0.0, 0.91, 0.91, 1.83])


print(fish2.mean())





# %%

import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7])

print(a.sum()) # 합 
print(a.mean()) # 평균
print(a.var()) # 분산
print(a.std()) # 표준편차
print(a.min())
print(a.max())
print(a.cumsum())
print(a.cumprod())




# %%

import numpy as np
import scipy.stats as ss

fish = np.array([2,3,3,4,4,4,4,5,5,6])
fish_s = (fish - fish.mean())/fish.std()

print('표준화 후 배열', fish_s)
print('표준화 후 평균', fish_s.mean())
print('표준화 후 표준편차', fish_s.std()) # np이용


# %%

import numpy as np
import scipy.stats as ss

fish = np.array([2,3,3,4,4,4,4,5,5,6])

fish_s = ss.zscore(fish) # ss 이용

print('표준화 후 배열', fish_s)
print('표준화 후 평균', fish_s.mean())
print('표준화 후 표준편차', fish_s.std())




# %%

import numpy as np

h = np.loadtxt('C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt')

print('길이: ',h.size)
print('평균: ',h.mean())
print('분산: ', h.var())
print('표준편차: ',h.std())



# %%

#25
a = (90-53.73)/29.85
b = (90-59.53)/22.97

print(a)
print(b)






# %%

a = (90-85)/5
b = (77-85)/5
print(a)
print(b)

a = (62-60)/2
b = (75-60)/2
print(a)
print(b)



# %%

import numpy as np

h = np.loadtxt('C:/Users/김보석/Desktop/대학/2024 - 1학기/확률및통계 01/중학생_남자_키.txt')

print('길이: ',h.size)
print('평균: ',h.mean())
print('분산: ', h.var())
print('표준편차: ',h.std())

a = (168 - h.mean())/h.std()
print('z-score: ', a)

# 0324
# %%

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt", sep='\t') # 텍스트 파일만 sep 필요

plt.hist(data, label='bins=10', bins=10) # 막대수10
plt.legend() # 범례
plt.show()

plt.hist(data, label='bins=5', bins=5)
plt.legend()
plt.show()

plt.hist(data, label='bins=100', bins=100)
plt.legend()
plt.show()







# %%

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt")

plt.hist(data, label='bins=10', bins=10, color='RED') # 막대수10
plt.legend() # 라벨 포함해서 보여줘
plt.show()

plt.hist(data, label='bins=5', bins=5)
#plt.legend()
plt.show()


# %%

from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt와 같은 의미
x = [6, 7, 8, 9, 10, 11]
y = [16109, 41401, 53121, 59899, 53450, 82565]
# x,y축 데이터 (리스트, 배열 모두 가능)
plt.bar(x, y)
plt.title('number of cases by month')
plt.show()


x = ['a', 'b', 'c', 'd', 'e', 'f'] # 문자 가능
y = [16109, 41401, 53121, 59899, 53450, 82565]
# x,y축 데이터 (리스트, 배열 모두 가능)
plt.bar(x, y)
plt.title('number of cases by month')
plt.show()


# %%

import matplotlib.pyplot as plt
ratio = [22, 24, 6, 38, 10] # 비율
labels = ['pizza', 'hamburger', 'pasta', 'chicken', 'bibimbab']
plt.pie(ratio, labels=labels, autopct='%.1f%%') # 소숫점 이하 첫째 자릿수 퍼센트로 구현
plt.title('asdfasdf')
plt.show() # 따옴표는 복붙할 시 서식 달라질 수 있음


# %%

x = [2014, 2015, 2016, 2017, 2018, 2019, 2020] # x축
y1 = [14.4, 14.5, 15.4, 16.9, 17.8, 17.6, 27.6] # 선 1
y2 =[20.5, 21.0, 22.8, 23.6, 24.2, 24.3, 29.5] # 선 2
plt.plot(x, y1, linestyle='solid', label='teens')
# x와 y1 그래프 작성 (직선)
plt.plot(x, y2, linestyle='dashed',label='20s')
# x와 y2 그래프 작성 (점선)
plt.legend(loc='best', ncol=2) 
# ncol은 범례표시 컬럼 수
plt.title('Internet Usage Time per Week')
plt.show()


# %%

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("C:/파이썬자료/초등학생_키몸무게/초등학생_키몸무게.xlsx")
plt.scatter(data.height, data.weight)
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

plt.scatter(data.weight, data.height)
plt.xlabel('weight')
plt.ylabel('height')
plt.show()


# %%

import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_excel("C:/파이썬자료/초등학생_키몸무게/초등학생_키몸무게.xlsx")

plt.hist(data, label='bins=6', bins=6)
plt.legend()
num = data['weight'].count() # 데이터 수

print(num)
print(1 + math.log2(num))  # 스터지스 공식에 의하면 7이 나온다

plt.hist(data['weight'], label='bins=7', bins=7)
plt.legend() # 범례
plt.show()


# %%


import pandas as pd
import numpy as np

x = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt") # 맨 위가 컬럼명으로 인식 : 컬럼명 141
y = np.loadtxt("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt")
print(x.size)
print(y.size)

x = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt", header=None) # 헤더 없으면 컬럼명 없이 맨 위부터 인식 데이터 141.0
print(x.size)
print(x.columns[0]) # 컬럼명 출력
x.columns=['height'] # 컬럼명 지정
print(x.size)
print(x.columns[0])
print(x.height[0])


# %%


#틀린거 찾기 2.5
import pandas as pd
import matplotlib as plt #1. matplotlib.pyplot
data = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt")
plt.bar(data, label='bins=10', bins=10) #2. plt.hist
plt.legend()

# %%


#틀린거 찾기 2
from matplotlib import pyplot #1. as plt
x = [6, 7, 8, 9, 10, 11]
y = [16109, 41401, 53121, 59899, 53450] #2. 숫자 하나 더 있어야 함
plt.bar(x, y, color='green')
plt.title('number of cases by month')
plt.show()


# %%


#틀린거 찾기 3
from matplotlib import pyplot as plt
x = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
y1 = [14.4, 14.5, 15.4, 16.9, 17.8, 17.6, 27.6]
y2 =[20.5, 21.0, 22.8, 23.6, 24.2, 24.3, 29.5]
plt.line(x, y1, linestyle='dashed', label='teens') # linestyle='solid’
# plt.plot
plt.line(x, y2, linestyle='solid', label='20s') # linestyle='dashed'
plt.legend(loc='best', ncol=2)
#plt.title()
plt.show()

# %%
#3
#import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/ ... /초등학생_키몸무게.xlsx") # .txt or read_excel
plt.scatter(data.weight, data.height) # 데이터 순서 반대로
plt.xlabel('height')
plt.ylabel('weight')
plt.show()


# %%


import numpy as np
import pandas as pd
from scipy import stats

x = pd.read_excel("C:/파이썬자료/중학생_남자_몸무게/중학생_남자_몸무게.xlsx", header = None)
print('mean평균 =', np.mean(x), '\n')
print('aver평균 =', np.average(x), '\n') # 가중치를 줄 수 있다
print('중앙값 =', np.median(x),'\n')
print('최빈값 =', stats.mode(x))


# %%


#확인문제 8
import numpy as np
a = np.array([[1, 2],
              [3, 4]] )
print('mean평균',np.mean(a))
print('axis=0일 때 평균',np.mean(a, axis=0))
#
#1 2
#3 4
#
#2 3 <- 이거 출력
#
print('axis=1일 때 평균',np.mean(a, axis=1))
#
#1 2   1.5
#3 4   3.5
#      위에거 출력
#
w = np.array([[0.1, 0.2],
              [0.3, 0.4]])
print('average', np.average(a))
print('가중평균',np.average(a, weights=w))
a = np.reshape(a, -1)
w = np.reshape(w, -1)
print('1차원 배열', a)
print('가중평균',np.average(a, weights=w))


# %%


import numpy as np

a = np.array([10,2,3,3,7,7,7,7,1,4])
print(np.median(a))
b = np.array([2,3,3,7,7,7,7,1,4])
print(np.median(b))

c = a.sort()

if (a.size % 2 == 0):
    print()
else:
    print()

    





#1. 1 2 3 3 4 7 7 7 7 10 -> 중앙값: 5.5
#2. 1 2 3 3 4 7 7 7 7 -> 중앙값: 4
#1. a.sort
#2. np.sort(a)


# %%


import numpy as np

a = np.array([10,2,3,3,7,7,7,7,1,4])
b = np.sort(a)
c = b.size

if (b.size % 2 != 0):
    result = b[c//2]

else:
    result = (b[c//2 - 1] + b[c//2]) / 2
print(b)
print(result)


# %%


import numpy as np
a = np.array([2,3,3,7,7,7,7,1,4])
b = np.sort(a)
c = b.size

if (b.size % 2 != 0):
    result = b[c//2]

else:
    result = (b[c//2 - 1] + b[c//2]) / 2
print(b)
print(result)



# %%


class Node:
  def __init__(self, item, next=None, prev=None):
    self.item = item
    self.next = None
    self.prev = None

class LinkedList:
  def __init__(self):
    self.head = None

  def insert(self, i:int, x:int):
    """insert x in ith element"""

  def delete(self, i):
    """delete ith element"""

  def printList(self):
    """여기에 코딩"""
    current = self.head
    while (current != None):
        print(current.item, end=" -> ")
        current = current.next
    print("end")
    

# %%

class Node:
  def __init__(self, item, next=None, prev=None):
    self.item = item
    self.next = None
    self.prev = None

class LinkedList:
  def __init__(self):
    self.head = None

  def insert(self, i:int, x:int):
    """insert x in ith element"""

  def delete(self, i):
    """delete ith element"""

  def printList(self):
    """여기에 코딩"""
    current = self.head
    while (current != None):
        print(current.item, end=" -> ")
        current = current.next
    print("end")
    

# %%

class Node:
    def __init__(self, item, next=None, prev=None):
        self.item = item
        self.next = next
        self.prev = prev

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, i: int, x: int):
        """insert x in ith element"""
        new_node = Node(x)
        if i == 0:
            new_node.next = self.head
            self.head = new_node
            return
        current = self.head
        for _ in range(i - 1):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next
        if current is None:
            raise IndexError("Index out of range")
        new_node.next = current.next
        current.next = new_node

    def delete(self, i):
        """delete ith element"""
        if i == 0:
            if self.head is None:
                raise IndexError("List is empty")
            self.head = self.head.next
            return
        current = self.head
        for _ in range(i - 1):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next
        if current is None or current.next is None:
            raise IndexError("Index out of range")
        current.next = current.next.next

    def printList(self):
        """Prints the linked list"""
        current = self.head
        while current != None:
            print(current.item, end=" -> ")
            current = current.next
        print("end")

# 테스트
if __name__ == "__main__":
    # 링크드 리스트 생성
    linked_list = LinkedList()
    # 노드 추가
    linked_list.insert(0, 1)
    linked_list.insert(1, 2)
    linked_list.insert(2, 3)
    linked_list.insert(3, 4)
    # 링크드 리스트 출력
    linked_list.printList()



# %%


class SingleLinkedList(LinkedList):
    def __init__(self):
        super().__init__()
        self.head = Node(None)
        self.dummy = Node(None, self.head)
        self.head.next = self.dummy

    def insert(self, i:int, x:int):
        newNode = Node(x)
        current = self.head
        while i != -1:
            current = current.next
            i -= 1
        newNode.next = current.next
        current.next = newNode

    def reverse(self):
        """여기에 코딩"""
        prev = self.head # 헤드를 이전 노드로 지정
        curr = prev.next # 헤드의 다음 노드를 현재 노드로 지정
        next = curr.next # 그 다음 노드를 다음 노드로 지정
        
        while(curr != self.dummy): # 현재 노드가 더미가 아닐 때 반복
            curr.next = prev # 현재 노드의 다음 노드를 이전 노드로 변경
            prev = curr # 이전 노드를 현재 노드로 변경
            curr = next # 현재 노드를 다음 노드로 변경
            next = next.next # 다음 노드를 다다음 노드로 변경 -> 한칸씩 다음 노드로 이동함
        self.head.next = prev # 현재 노드가 더미 노드일 때(헤드 일 때) 다음 노드를 이전 노드로 변경
        
            
# %%


import numpy as np
import pandas as pd
from scipy import stats


a = np.array([1,2,3,3,4,4,6,7,7,10])
print('a의 mean =',a.mean())
print('a의 10% 절사평균 =', stats.trim_mean(a, 0.1)) # 0.1은 양쪽에서 10퍼씩 제외한다 총 20퍼
print('\n')
x = pd.read_excel("C:/파이썬자료/중학생_남자_몸무게/중학생_남자_몸무게.xlsx", header=None) # 헤더 넣어서 0 출력
print('x의 mean =',x.mean()) # header 없어서 47.5가 헤더로 출력
print('x의 10% 절사평균 =', stats.trim_mean(x, 0.1))


# %%


import pandas as pd
from scipy import stats

data = pd.DataFrame(pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx"))
#print("왜도 : ", stats.skew(data)) # 0은 첫번째인 지역이다 의미x
#print("첨도 : ", stats.kurtosis(data))
#print("\n\n\n")
print(data.describe())
print()
print(stats.describe(data)) # 참고로 알고있어라


# %%

# 확인문제 17
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.DataFrame(pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx"))
plt.hist(data.냉면, label='bins=5', bins=5)
plt.legend() # 범례
plt.show()


plt.hist(data.삼계탕, label='bins=5', bins=5, color='green')
plt.legend() # 범례
plt.show()

plt.hist(data.김밥, label='bins=5', bins=5, color='orange')
plt.legend() # 범례
plt.show()

print("왜도 : ", stats.skew(data.냉면))
print("왜도 : ", stats.skew(data.삼계탕))
print("왜도 : ", stats.skew(data.김밥))


# %%


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.DataFrame(pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx"))
plt.hist(data.냉면, label='bins=5', bins=5)
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.냉면)) # 0.161 4
print()
print()

plt.hist(data.비빔밥, label='bins=5', bins=5, color='green')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.비빔밥)) # 0.612 2
print()
print()

plt.hist(data.김치찌개, label='bins=5', bins=5, color='orange')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.김치찌개)) # 1.002 1
print()
print()

plt.hist(data.삼겹살, label='bins=5', bins=5, color='red')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.삼겹살)) # 0.039 5
print()
print()

plt.hist(data.자장면, label='bins=5', bins=5, color='yellow')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.자장면)) # 0.167 3
print()
print()

plt.hist(data.삼계탕, label='bins=5', bins=5, color='purple')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.삼계탕)) # -0.723 8    
print()
print()

plt.hist(data.칼국수, label='bins=5', bins=5, color='coral')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.칼국수)) # -0.048 6
print()
print()

plt.hist(data.김밥, label='bins=5', bins=5, color='crimson')
plt.legend() # 범례
plt.show()
print("왜도 : ", stats.skew(data.김밥)) # -0.128 7
print()  
print()


# %%


import pandas as pd
import matplotlib.pyplot as plt

file_data = pd.read_csv("C:/파이썬자료/sample/sample1.csv") # 데이터 프레임
print(file_data[0:5]) # 맨 위 다섯 행만 보겠다 - 일부 출력
              

#                시리즈                  시리즈
total_score = file_data['점수'] * 5 + file_data['출석'] # 시리즈
print(type(total_score)) # 시리즈 타입 출력



#           시리즈
new_data = [file_data['이름'], total_score] # 리스트에 넣음
print(type(new_data)) # 리스트 타입 출력


#         합치기                      컬럼명                     
result = pd.concat(new_data, axis=0, keys=['name', 'total']) # axis = 0: 위 아래, 1: 왼쪽 오른쪽
print(type(result)) # 데이터 프레임
print(result) #  
result.to_excel("C:/파이썬자료/psdata/result1.xlsx") # 파일 만들기
#     파일 만들기


plt.hist(total_score, label='score data', bins=7) # 
plt.legend() # 
plt.savefig("C:/파이썬자료/psdata/histogram of score.png") # 그림 저장
plt.show() # 


# %%


#19번 문제 과제
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")

new_data = [data['냉면'], data['비빔밥']] # 외식비 파일의 냉면과 비빔밥만 읽어오기
result = pd.concat(new_data, axis=0, keys=['냉면', '비빔밥']) # 냉면과 비빔밥 합침
print(result) # 확인용 출력

plt.hist(result, label='bins=6', bins=6) # 스터지수공식에 의한 막대 갯수 6개로 히스터그램 만들기
plt.legend() # 범례
plt.savefig("C:/파이썬자료/psdata/19번 과제 냉면비빔.png") # png파일로 저장
plt.show()


# %%


# 20번 과제
import pandas as pd
import matplotlib.pyplot as plt

weight = pd.read_excel("C:/파이썬자료/중학생_남자_몸무게/중학생_남자_몸무게.xlsx", header=None) # 헤더 없음
height = pd.read_excel("C:/파이썬자료/중학생_남자_키/중학생_남자_키.xlsx", header = None) # 헤더 없음

new_data = [weight, height] # 두 데이터를 합침
result = pd.concat(new_data, axis = 1, keys=['몸무게', '키']) # 2열로 합침
print(result) # 확인용 프린트

plt.scatter(result['몸무게'], result['키']) # 산점도로 바꿈
plt.xlabel('weight') # x축은 몸무게
plt.ylabel('height') # y축은 키
plt.title('asdfasdf')
plt.savefig("C:/파이썬자료/psdata/20번 과제 산점도.png") # png파일로 산점도 그림 저장
plt.show() # 보여주기


# %%


import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 1, 2, 3, 4]) # 확률 변수 x의 값
count = np.array([5, 25, 40, 25, 5]) # 빈도
prob = count/100 # np.array는 숫자로 나누면 모든 항목이 나눠진다
plt.bar(x,prob)
plt.show()


# %%


import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5]) # 확률 변수 x의 값
count = np.array([8, 12, 15, 30, 85]) # 빈도
prob = count/150 # np.array는 숫자로 나누면 모든 항목이 나눠진다
plt.bar(x,prob)
plt.show()


# %%


# 동전 두번 던지기
import numpy as np
import matplotlib.pyplot as plt

arr = np.array = [0, 1, 2] # 앞면이 나오는 확률변수 값 (확률변수 X)
parr = np.array[0.25, 0.5, 0.25] # 확률 (확률 P(X))
plt.bar(arr, parr)
plt.xlabel('X')
plt.ylabel('P(X)')
plt.show()


# %%


#동전 세개 던지기
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([0, 1, 2, 3]) # 앞면이 나오는 확률변수 값 (확률변수 X)
parr = np.array([1, 3, 3, 1]) # 빈도수
prob = parr / 8
plt.bar(arr, prob)
plt.xlabel('X')
plt.ylabel('P(X)')
plt.show()


# %%


import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5]) # 확률 변수 x의 값
count = np.array([8, 12, 15, 30, 85]) # 빈도
prob = count / 150 # np.array는 숫자로 나누면 모든 항목이 나눠진다
plt.bar(x,prob)
plt.show()


# %%


import numpy as np


def stusis(num):
    return np.ceil(1 + np.log2(num))
    
def csvSample():
    return "file_df = pd.read_csv(\"C:/'''/sample3.txt\", sep='\t', encoding=\"UTF-8\")"

def excelSample():
    return "file_df = pd.read_excel(\"C:/'''/sample4.xlsx\")"




# %%

# 4장 확인문제 4.
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
n = 3 # 횟수
p = 1 / 3 # 성공 확률
x = np.arange(n+1)
mean, var = binom.stats(n, p)
prob = binom.pmf(x, n, p)
print('mean=', mean, 'var=', var)
plt.bar(x, prob)
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('binomial distribution(n=3, p=1/3)')
plt.show()
print(prob)


# %%


a = [1, 2, 3, 4, 5]

a.extend(['989'])
print(a)



         

# %%
    
class ListNode:
    def __init__(self, newItem, nextNode = 'ListNode'):
        self.item = newItem
        self.next = nextNode

# %%

class LinkedListBasic:
    def __init__(self):
        self.__head = ListNode('dummy', None)
        self.__numItems = 0

    def insert(self, i, newItem):
        if i >= 0 and i <= self.__numItems:
            prev = self.__getNode(i - 1)
            newNode = ListNode(newItem, prev.next)
            prev.next = newNode
            self.__numItems += 1
        else:
            print("에러 ")
        
    def append(self, newItem):
        prev = self.__getNode(self.__numItems - 1)
        newNode = ListNode(newItem, prev.next)
        prev.next = newNode
        self.__numItems += 1

    def pop(self, i):
        if i >= 0 and i <= self.__numItems - 1:
            prev = self.__getNode(i - 1)
            curr = prev.next
            prev.next = curr.next
            getitem = curr.item
            self.__numItems -= 1
            return getitem
        else:
            return None
        
    def remove(self, item):
        (prev, curr) = self.__fineNode(item)
        if curr != None:
            prev.next = curr.next
            self.__numItems -= 1
            return item
        else:
            return None
    
    def get(self, i):
        if self.isEmpty():
            return None
        if i >= 0 and i <= self.__numItems - 1:
            return self.__getNode(i).item
        else:
            return None

    def index(self, findItem):
        curr = self.__head.next
        for index in range(self.__numItems):
            if curr.item == findItem:
                return index
            else:
                curr = curr.next
        return None
                    
    def isEmpty(self):
        return self.__numItems == 0
        
    def size(self):
        return self.__numItems

    def clear(self):
        self.__head = ListNode('dummy', None)
        self.__numItems = 0

    def count(self, x):
        cnt = 0
        curr = self.__head.next
        for index in range(self.__numItems):
            if curr.item == x :
                cnt += 1
            curr = curr.next
        return cnt

    def extend(self, a):
        for index in range(a.size()):
            self.append(a.get(index))

    def copy(self):
        a = LinkedListBasic()
        for index in range(self.__numItems):
            a.append(self.get(index))
        return a
        
    def reverse(self):
        a = LinkedListBasic()
        for index in range(self.__numItems):
            a.insert(0, self.get(index))
        self.clear()
        for index in range(a.size()):
            self.append(a.get(index))

    def sort(self):
        a = []
        for index in range(self.__numItems):
            a.append(self.get(index))
            a.sort()
            self.clear()
        for index in range(len(a)):
            self.append(a[index])

    def __findNode(self, x):
        prev = self.__head
        curr = prev.next
        while curr != None:
            if curr.item == x:
                return (prev, curr)
            else:
                prev = curr
                curr = curr.next
        return (None, None)
        
    def __getNode(self, i):
        curr = self.__head.next
        for index in range(i + 1):
            curr = curr.next
        return curr
            
    def printList(self):
        curr = self.__head.next
        while curr != None:
            print(curr.item, end = " ")
            curr = curr.next
        print()    
        





# %%

list = LinkedListBasic()
list.append(30)
list.insert(0, 20)

a = LinkedListBasic()
a.append(4)
a.append(3)
a.append(3)
a.append(2)
a.append(1)

list.extend(a)
list.reverse()
list.pop(0)

print("count(3):", list.count(3))
print("get(2):", list.get(2))
list.printList()








# %%

class LinkedListBasic:
	def __init__(self):
		self.__head = ListNode('dummy', None)
		self.__numItems = 0

	# [알고리즘 5 - 2] 구현: 연결 리스트에 원소 삽입하기(더미 헤드를 두는 대표 버전)
	def insert(self, i:int, newItem):
		if i >= 0 and i <= self.__numItems:
			prev = self.__getNode(i - 1)
			newNode = ListNode(newItem, prev.next)
			prev.next = newNode
			self.__numItems += 1
		else:
			print("index", i, ": out of bound in insert()") # 필요 시 에러 처리

	def append(self, newItem):
		prev = self.__getNode(self.__numItems - 1)
		newNode = ListNode(newItem, prev.next)
		prev.next = newNode
		self.__numItems += 1

	# [알고리즘 5-3] 구현: 연결 리스트의 원소 삭제하기
	def pop(self, i:int):   # i번 노드 삭제. 고정 파라미터
		if (i >= 0 and i <= self.__numItems-1):
			prev = self.__getNode(i - 1)
			curr = prev.next
			prev.next = curr.next
			retItem = curr.item
			self.__numItems -= 1
			return retItem
		else:
			return None

	# [알고리즘 5 -4] 구현: 연결 리스트의 원소 x 삭제하기 (더미 헤드를 두는 대표 버전)
	def remove(self, x):
		(prev, curr) = self.__findNode(x)
		if curr != None:
			prev.next = curr.next
			self.__numItems -= 1
			return x
		else:
			return None

	# [알고리즘 5 - 5] 구현: 연결 리스트의 i번 원소 알려주기
	def get(self, i:int):
		if self.isEmpty():
			return None
		if (i >= 0 and i <= self.__numItems - 1):
			return self.__getNode(i).item
		else:
			return None

	# [알고리즘 5 -7] 구현: x가 연결 리스트의 몇 번 원소인지 알려주기
	def index(self, x) -> int:
		curr = self.__head.next	 # 0번 노드:  더미 헤드 다음 노드
		for index in range(self.__numItems):
			if curr.item == x:
				return index
			else:
				curr = curr.next
		return -2 # 안 쓰는 인덱스

	# [알고리즘 5 -8] 구현: 기타 작업들
	def isEmpty(self) -> bool:
		return self.__numItems == 0

	def size(self) -> int:
		return self.__numItems

	def clear(self):
		self.__head = ListNode("dummy", None)
		self.__numItems = 0

	def count(self, x) -> int:
		cnt = 0
		curr = self.__head.next  # 0번 노드
		while curr != None:
			if curr.item == x:
					cnt += 1
			curr = curr.next
		return cnt

	def extend(self, a): # 여기서 a는 self와 같은 타입의 리스트
		for index in range(a.size()):
			self.append(a.get(index))

	def copy(self):
		a = LinkedListBasic()
		for index in range(self.__numItems):
			a.append(self.get(index))
		return a

	def reverse(self):
		a = LinkedListBasic()
		for index in range(self.__numItems):
			a.insert(0, self.get(index))
		self.clear()
		for index in range(a.size()):
			self.append(a.get(index))

	def sort(self) -> None:
		a = []
		for index in range(self.__numItems):
			a.append(self.get(index))
		a.sort()
		self.clear()
		for index in range(len(a)):
			self.append(a[index])

	def __findNode(self, x):
		prev = self.__head  # 더미 헤드
		curr = prev.next    # 0번 노드
		while curr != None:
			if curr.item == x:
				return (prev, curr)
			else:
				prev = curr; curr = curr.next
		return (None, None)

	# [알고리즘 5-6] 구현: 연결 리스트의 i번 노드 알려주기
	def __getNode(self, i:int) -> ListNode:
		curr = self.__head # 더미 헤드, index: -1
		for index in range(i+1):
			curr = curr.next
		return curr

	def printList(self):
		curr = self.__head.next # 0번 노드: 더미 헤드 다음 노드
		while curr != None:
			print(curr.item, end = ' ')
			curr = curr.next
		print()



# %%


class BidirectNode:
    def __inif__(self, x, prevNode : 'BidirectNode', nextNode : 'BidirectNode'):
        self.item = x
        self.prev = prevNode
        self.next = nextNode









# %%

count = 0
count2 = 0
def move(n):
    global count, count2
    count += 1
    if ( n > 0 ):
        move(n-1)
        count2 += 1
        move(n-1)

move(4)
print(count, count2)





# %%


a = 3
b = 7
print(a, b)

a, b = b, a

print(a, b)








# %%


a = [5, 2, 6, 2 , 1, 3]
a.pop(0)
a.insert(0, 99)
a.append(55)
print(a)

print(len(a))





# %%

import numpy as np
a = np.array([[1,2,3,4,5],
             [6,7,8,9,10]])
print(a[1, 2:4])


# %%

import numpy as np
import pandas as pd

file = pd.read_csv("C:/파이썬자료/sample/sample2.txt",  sep='\t', encoding="UTF-8")

print(file)










# %%
import numpy as np
import pandas as pd


df1 = pd.DataFrame({
    'col1' : np.array([1,2,3]),
    'col2' : np.array(['A','B','C'])
})
df2= pd.DataFrame({
    'col1' : np.array([4,5,6]),
    'col2' : np.array(['D','E','F'])
})
df3= pd.DataFrame({
    'col1' : np.array([4,5,6]),
    'col2' : np.array(['D','E','F'])
})

df = pd.DataFrame({
'col1' : [1,2,3,4,5],
'col2' : [2,4,6,8,10],
'col3' : ['A','B','C','D','E'] })

print(df.query('col1 >= 3')) 




# %%

import matplotlib.pyplot as plt
ratio = [22, 24, 16, 50] # 비율
labels = ['pizza', 'hamburger', 'pasta', 'chicken']
plt.pie(ratio, labels=labels, autopct='%.1f%%')
plt.show()


# %%


from matplotlib import pyplot as plt
x = [2014, 2015, 2016, 2017, 2018, 2019, 2020] # x축
y1 = [14.4, 14.5, 15.4, 16.9, 17.8, 17.6, 27.6] # 선 1
y2 =[20.5, 21.0, 22.8, 23.6, 24.2, 24.3, 29.5] # 선 2
plt.plot(x, y1, linestyle='solid', label='teens')
# x와 y1 그래프 작성 (직선)
plt.plot(x, y2, linestyle='dashed',label='20s')
# x와 y2 그래프 작성 (점선)
plt.legend(loc='best', ncol=2)
# ncol은 범례표시 컬럼 수
plt.title('Internet Usage Time per Week')
plt.show()


# %%


import numpy as np
import pandas as pd
from scipy import stats
x = pd.read_excel("C:/파이썬자료/중학생_남자_몸무게/중학생_남자_몸무게.xlsx")
print('평균 =', np.mean(x), '\n')
print('중앙값 =', np.median(x),'\n')
print('최빈값 =', stats.mode(x))








# %%


import pandas as pd
from scipy import stats
data = pd.DataFrame(pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx"))
print(data.describe())
print()
print(stats.describe(data))


# %%

import numpy as np
import pandas as pd

arr1 = np.array([1, 2, 3, 4 ,5])
arr2 = np.array([6, 7, 8, 9, 10])


print(arr1)
print(arr1+arr2)
print(arr1+5)
print(arr1-arr2)
print(arr1-5)
print(arr1*arr2)
print(arr1*2)
print(arr1/arr2)
print(arr1/5)


# %%

import numpy as np
import pandas as pd

list1 = (1, 2, 3, 4, 5)
list2 = (6, 7, 8, 9, 10)

print(list1)
print(list1+list2)
#print(list1+5)
#print(list1-list2)
#print(list1-5)
#print(list1*list2)
print(list1*2)
#print(list1/list2)
#print(list1/5)


# %%

import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.add(a,b) #c=a+b
print(c)
d = np.dot(a,b)
print(d)


# %%

import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
print('합:', fish.sum(), np.sum(fish))
print('개수:', fish.size, np.size(fish))
print('평균:',fish.mean(), np.mean(fish))


# %%


import numpy as np
a = np.arange(20)
print(a)
print(a.reshape(4,5))
print(a)






# %%


import pandas as pd
df1 = pd.DataFrame({
 'a' : [1,2,3],
 'b' : [4,5,6]
 })
df2 = pd.DataFrame({
 'c' : [7,8,9],
 'd' : [10,11,12]
 })
cc = pd.concat([df1, df2], axis=0)
print(cc, '\n')
cc = pd.concat([df1, df2], axis=1)
print(cc, '\n')








# %%


import pandas as pd
a = pd.DataFrame(
 [['A', 1], ['B', 2], ['C', 3]],
 columns=['x1', 'x2']
 )
b = pd.DataFrame({
 'x1':['B','C','D'],
 'x3':[2,3,4]
 })
print(a,'\n')
print(b,'\n')
print(pd.concat([a,b]), '\n')
print(pd.concat([a,b], axis=1), '\n')


# %%


import numpy as np
import pandas as pd
df1 = pd.DataFrame({
'col1' : np.array([1,2,3]),
'col2' : np.array(['A','B','C'])
})
df2= pd.DataFrame({
'col1' : np.array([4,5,6]),
'col2' : np.array(['D','E','F'])
})
print(pd.concat([df1,df2], axis = 1), '\n')


# %%


import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])

fish1 = (fish - fish.min()) / fish.max()
fish2 = (fish - fish.mean()) / fish.std()

print(fish2)
print(fish2.std())




# %%


import numpy as np
import scipy.stats as ss

fish = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 6])

# 데이터를 표준화
fishs = ss.zscore(fish)

print("표준화된 데이터:", fishs)








# %%

import numpy as np

asdf = np.loadtxt("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt")



print("데이터 수:", len(asdf))
print("평균:", np.mean(asdf))
print("분산:", np.var(asdf))
print("표준편차:", np.std(asdf))







# %%

import scipy.stats as stats

# 도깨비와 저승이의 점수
score_dokkaebi = 90
score_jeoseung = 90

# 평균과 표준편차
mean_dokkaebi, std_dokkaebi = 53.73, 29.85
mean_jeoseung, std_jeoseung = 59.53, 22.97

# 도깨비와 저승이의 점수를 표준 정규 분포로 변환
z_score_dokkaebi = (score_dokkaebi - mean_dokkaebi) / std_dokkaebi
z_score_jeoseung = (score_jeoseung - mean_jeoseung) / std_jeoseung

# 백분위 순위 계산
percentile_dokkaebi = stats.norm.cdf(z_score_dokkaebi) * 100
percentile_jeoseung = stats.norm.cdf(z_score_jeoseung) * 100

# 결과 출력
print("도깨비의 백분위 순위:", percentile_dokkaebi)
print("저승이의 백분위 순위:", percentile_jeoseung)










# %%
import pandas as pd
import numpy as np

x = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt", header=None)
print(x[0:4])
print(x.size)
print(x.columns[0])
x.columns=['height']
print(x.height[0])









# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")

nm = plt.hist(x.냉면, label = "nm", bins = "auto")
plt.legend()
plt.show()

sam = plt.hist(x.삼계탕, label = "sam", bins = "auto")
plt.legend()
plt.show()

gim = plt.hist(x.김밥, label = "gim", bins = "auto")
plt.legend()
plt.show()




# %%


# 각 행의 의미를 설명하세요.
import pandas as pd
import matplotlib.pyplot as plt
file_data = pd.read_csv("C:/파이썬자료/sample/sample1.csv")
print(file_data[0:5])
total_score = file_data['점수'] * 5 + file_data['출석']

print(type(total_score))
print(total_score)
new_data = [file_data['이름'], total_score]
print(type(new_data))
result = pd.concat(new_data, axis=1, keys=['name', 'total'])
print(type(result))
print(result)
result.to_excel("C:/파이썬자료/sample/result1.xlsx")
plt.hist(total_score, label='score data', bins=7)
plt.legend()
plt.savefig("C:/파이썬자료/sample/histogram of score.png")
plt.show()


# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")

result = pd.concat((data.냉면, data.비빔밥), axis=0, keys=['냉면', '비빔밥'])
print(result.size)
print(result)

plt.hist(result, label = 'bins = 6', bins = 6)
plt.legend()
plt.savefig("C:/sabe.png")
plt.show()



# %%


import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
print(df.columns)
df.columns = ['area','nang','bibim','kimchi','samsal','jja','samtang','kal','kimbob']
print(df[0:2])


# %%


import numpy as np
import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
df = df.drop('지역', axis=1)
print(df[0:3])
print()
print('데이터프레임 최대값 df.max() ==>')
print(df.max())
print()
print('넘파이 최대값 np.max(df) ==>')
print(np.max(df))
print('넘파이 최대값 np.max(df, axis=0) ==>')
print(np.max(df, axis=0))


# %%


import numpy as np
import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
print('df 냉면 평균 --> ', df.냉면.mean())
print('df 냉면 분산 --> ', df.냉면.var())
print()
print('np 냉면 평균 --> ', np.mean(df.냉면))
print('np 냉면 분산 --> ', np.round(np.var(df.냉면, ddof = 1), 2))

print(len(df))
print(df.size)
print(np.size(df))


# %%



# %%







# %%


import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
print('분산:',fish.var(), np.var(fish) )
print('불편분산:',fish.var(ddof=1), np.var(fish, ddof=1) )
print('표준편차:',fish.std(), np.std(fish) )
print('불편표준편차:',fish.std(ddof=1), np.std(fish, ddof=1) )



















# %%


import numpy as np
from scipy.stats import binom

# 시행 횟수
n = 3
# 성공 확률
p = 1/3

# 이항분포 생성
X_distribution = binom(n, p)

# 기댓값
expected_value = X_distribution.mean()

# 분산
variance = X_distribution.var()

print("기댓값:", expected_value)
print("분산:", variance)











# %%
import pandas as pd
import numpy as np

x = pd.read_csv("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt", header=None)
print(x[0:4])
print(x.size)
print(x.columns[0])
x.columns=['height']
print(x.height[0])









# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")

nm = plt.hist(x.냉면, label = "nm", bins = "auto")
plt.legend()
plt.show()

sam = plt.hist(x.삼계탕, label = "sam", bins = "auto")
plt.legend()
plt.show()

gim = plt.hist(x.김밥, label = "gim", bins = "auto")
plt.legend()
plt.show()




# %%


# 각 행의 의미를 설명하세요.
import pandas as pd
import matplotlib.pyplot as plt
file_data = pd.read_csv("C:/파이썬자료/sample/sample1.csv")
print(file_data[0:5])
total_score = file_data['점수'] * 5 + file_data['출석']

print(type(total_score))
print(total_score)
new_data = [file_data['이름'], total_score]
print(type(new_data))
result = pd.concat(new_data, axis=1, keys=['name', 'total'])
print(type(result))
print(result)
result.to_excel("C:/파이썬자료/sample/result1.xlsx")
plt.hist(total_score, label='score data', bins=7)
plt.legend()
plt.savefig("C:/파이썬자료/sample/histogram of score.png")
plt.show()


# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")

result = pd.concat((data.냉면, data.비빔밥), axis=0, keys=['냉면', '비빔밥'])
print(result.size)
print(result)

plt.hist(result, label = 'bins = 6', bins = 6)
plt.legend()
plt.savefig("C:/sabe.png")
plt.show()



# %%


import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
print(df.columns)
df.columns = ['area','nang','bibim','kimchi','samsal','jja','samtang','kal','kimbob']
print(df[0:2])


# %%


import numpy as np
import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
df = df.drop('지역', axis=1)
print(df[0:3])
print()
print('데이터프레임 최대값 df.max() ==>')
print(df.max())
print()
print('넘파이 최대값 np.max(df) ==>')
print(np.max(df))
print('넘파이 최대값 np.max(df, axis=0) ==>')
print(np.max(df, axis=0))


# %%


import numpy as np
import pandas as pd
df = pd.read_excel("C:/파이썬자료/외식비/외식비.xlsx")
print('df 냉면 평균 --> ', df.냉면.mean())
print('df 냉면 분산 --> ', df.냉면.var())
print()
print('np 냉면 평균 --> ', np.mean(df.냉면))
print('np 냉면 분산 --> ', np.round(np.var(df.냉면, ddof = 1), 2))

print(len(df))
print(df.size)
print(np.size(df))


# %%


import pandas as pd
df1 = pd.DataFrame({
 'a' : [1,2,3],
 'b' : [4,5,6]
 })
df2 = pd.DataFrame({
 'a' : [7,8,9],
 'b' : [10,11,12]
 })
cc = pd.concat([df1, df2], axis=0)
print(cc, '\n')
cc = pd.concat([df1, df2], axis=1)
print(cc, '\n')








# %%


import numpy as np
fish = np.array([2,3,3,4,4,4,4,5,5,6])
print('분산:',fish.var(), np.var(fish) )
print('불편분산:',fish.var(ddof=1), np.var(fish, ddof=1) )
print('표준편차:',fish.std(), np.std(fish) )
print('불편표준편차:',fish.std(ddof=1), np.std(fish, ddof=1) )









# %%


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
n=3
p=1/3
x = np.arange(n+1)
mean, var = binom.stats(n, p)
prob = binom.pmf(x, n, p)
print('mean=', mean, 'var=', var)
plt.bar(x, prob)
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('binomial distribution(n=5, p=1/3)')
plt.show()
print(prob)








# %%


import numpy as np
from scipy.stats import binom

# 시행 횟수
n = 3
# 성공 확률
p = 1/3

# 이항분포 생성
X_distribution = binom(n, p)

# 기댓값
expected_value = X_distribution.mean()

# 분산
variance = X_distribution.var()

print("기댓값:", expected_value)
print("분산:", variance)










# %%



# 7  3
# <3, 6, 2, 7, 5, 1, 4>
a, b = map(int, input().split())
yose = list(range(1, a + 1))
print(yose)
num = 0
print(a, b)


while(len(yose) != 0):
    num = (num + b - 1) % a # a 하면 안됨
    print(yose.pop(num))


# %%


# 7  3
# <3, 6, 2, 7, 5, 1, 4>
a, b = map(int, input().split())
yose = list(range(1, a + 1))

num = 0
print("<", end = "")


while(len(yose) != 0):
    num = (num + b - 1) % len(yose)  # 리스트의 길이로 나눈 나머지를 사용합니다.
    if (len(yose) == 1): print(yose.pop(num), end = "")
    else: print(yose.pop(num), end = ", ")
print(">")


# %%


a = int(input())

priQ = list()

while(a > 0):
    b = int(input())
    if(b == 0):
        if(len(priQ) == 0): print(0)
        else:
            minV = min(priQ)
            print(minV)
            priQ.remove(minV)
    else: priQ.append(b)
    a = a - 1


# %%


from scipy.stats import norm
mu=0              # 평균
sigma=1           # 표준편차
# p. 26의 binorm.cdf 와 다름
y1 = norm.cdf(0.5, mu, sigma)
y2 = norm.cdf(-1.5, mu, sigma)
print('P(-1.5≤Z≤0.5)=', y1-y2)


# %%


#둘리 마루 z = 1.1, 1.2
from scipy.stats import norm
mu=0              # 평균
sigma=1           # 표준편차
# p. 26의 binorm.cdf 와 다름
dooli = norm.cdf(1.1, mu, sigma)
maru= norm.cdf(1.2, mu, sigma)
print('dooli cdf =' , dooli)
print('maru cdf =' , maru)

print('dooli 상위% =', round((1 - dooli)*100, 2))
print('maru 상위% =', round((1 - maru)*100, 2))


# %%


from scipy.stats import norm
mu=0
sigma=1
percent_point=0.9
print('P(Z≤k)=0.9, k=',norm.ppf(percent_point, mu, sigma))


# %%


# 4장 8번
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
mu = 7
sigma = 15

x = np.arange(mu-50, mu+50, 0.1)
y = norm.pdf(x, mu, sigma)

plt.bar(x, y)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('normal distribution(mu=7, sigma=15)')
plt.show()

mu = 175
sigma = 20

x = np.arange(mu-50, mu+50, 0.1)
y = norm.pdf(x, mu, sigma)

plt.bar(x, y, color='green')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('normal distribution(mu=175, sigma=20)')
plt.show()


# %%


#4장 10번
from scipy.stats import norm
mu=0              # 평균
sigma=1           # 표준편차
y1 = norm.cdf(-1.5, mu, sigma)
y2 = norm.cdf(1.5, mu, sigma)
print(y1 - y2)




# %%


#4장 11번
from scipy.stats import norm
mu=75              # 평균
sigma=10           # 표준편차
print(norm.cdf(80, mu, sigma) - norm.cdf(90, mu, sigma))








# %%


#4장 13번
from scipy.stats import norm
mu=0 # 표준화 하면 평균 0, 표편 1
sigma=1
percent_point = 0.1
print(norm.ppf(percent_point, mu, sigma))


# %%


#4장 14번
from scipy.stats import norm
mu=300000
sigma=100000

print(norm.cdf(350000, mu, sigma) - norm.cdf(250000, mu, sigma))


# %%


#4장 15번
from scipy.stats import norm
mu=70
sigma=10

print(1 - norm.cdf(95, mu, sigma))
print(1500 * 0.0062, "명")


# %%


#4장 16번
from scipy.stats import norm
mu=75 
sigma=10
percent_point = 0.95
print(norm.ppf(percent_point, mu, sigma))


# %%


import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import norm

plt.rcParams['font.family'] = 'Malgun Gothic'
data = np.loadtxt("C:/파이썬자료/중학생_남자_키/중학생_남자_키.txt")
N = len(data) # 4933
n = 30 # n>=30
trial = 200 # n개짜리 표본을 200번 추출 (충분히 여러 번 해줌)
x = np.zeros(n) #표본 1개 (길이 30)
xe = np.zeros(trial) #표본평균값들을 저장한 배열 (길이 200)
for k in range(trial) :
    for i in range(n) : 
         x[i] = data[random.randint(0,N-1)]
    xe[k] = np.mean(x) # 랜덤이므로 실행할 때마다 결과 조금씩 다름
print('모집단', np.mean(data), np.var(data))
print('표본평균',np.mean(xe), np.var(xe, ddof=1))
ye = norm.pdf(xe, np.mean(xe), np.std(xe, ddof=1))
plt.bar(xe, ye)
plt.xlabel('표본평균 n='+str(n)); plt.ylabel('확률')
plt.show()


# %%


import numpy as np

x = np.array([8, 9, 8, 10, 10])
print("표본 평균: ", np.mean(x))
print("표본 분산: ", np.var(x))
print("표본불편 분산: ", np.var(x, ddof = 1))
# 추정 모집단 분산


# %%


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t # t 분포

df=2 # 자유도
x = np.arange(-4, 4, 0.01)
y = t.pdf(x, df)
plt.bar(x, y)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('t distribution(df=2)')
plt.show()


# %%


from scipy.stats import t
from scipy.stats import norm

df=2
y1 = t.cdf(2, df)
y2 = t.cdf(-2, df)
print('t분포 P(-2≤t≤2)=', y1-y2)

# 정규분포
mu=0
sigma=1
y1 = norm.cdf(2, mu, sigma)
y2 = norm.cdf(-2, mu, sigma)
print('표준정규분포 P(-2≤Z≤2)=', y1-y2)


# %%


from scipy.stats import t
from scipy.stats import norm
df=2
percent_point=0.95
print('t분포 P(t≤k)=0.95일 때 k=', t.ppf(percent_point, df))
mu=0
sigma=1
print('표준정규분포 P(Z≤k)=0.95일 때 k=', norm.ppf(percent_point, mu, sigma))


# %%


#5장 확인문제 4
from scipy.stats import t
from scipy.stats import norm


print("자유도 10: ", t.cdf(1, df = 10) - t.cdf(-1, df = 10)) # 1부터 -1까지 범위
print("자유도 30: ", t.cdf(1, df = 30) - t.cdf(-1, df = 30))
print("자유도 50: ", t.cdf(1, df = 50) - t.cdf(-1, df = 50))
print("자유도 100: ", t.cdf(1, df = 100) - t.cdf(-1, df = 100)) # 점점 커진다

mu = 0
sigma = 1
print("표준정규분포: ", norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)) # 얘가 가장 큼



# %%


#5장 확인문제 5
from scipy.stats import t
from scipy.stats import norm


print("자유도 100: ", t.ppf(0.95, df = 100))

mu = 0
sigma = 1
print("표준정규분포: ", norm.ppf(0.95, mu, sigma)) # 얘가 가장 큼


# %%


#5장 확인문제 6
#np.random.normal(평, 표, size) 얘 많이 사용
#random.sample(range(20, 120), 10) 중복 불가
#random.choices(range(20, 120), k=1000) 중복 가능
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(70, 15, size = 1000)

plt.hist(x, bins = 11)
plt.show()


# %%


from scipy.stats import norm
mu = 0
sigma = 1
print(norm.ppf(0.975, mu, sigma))


# %%


# 임시 예제 14
from scipy.stats import norm
mu = 0
sigma = 1
print(norm.ppf(0.975, mu, sigma)) # 95%
print(norm.ppf(0.995, mu, sigma)) # 99%


# %%


# 예제 14 정규분포
from scipy.stats import norm
from scipy.stats import t
import numpy as np
mu = 70.4
n = 50
s = 6
sem = s / np.sqrt(n)
zu = norm.ppf(0.975, 0, 1) # 95%

print('하한: ', mu - zu * sem)
print('상한: ', mu + zu * sem)

print(norm.interval(0.95, loc = mu, scale = sem))

zu = norm.ppf(0.995, 0, 1) # 99%

print('하한: ', mu - zu * sem)
print('상한: ', mu + zu * sem)

print(norm.interval(0.99, loc = mu, scale = sem))
print('t분포: ', t.interval(0.99, df = 49, loc = mu, scale = sem))


# %%


# t 분포 5.2 20p
import numpy as np
from scipy.stats import t
from scipy import stats
# from scipy.stats import sem
a = [260,265, 250,270, 272, 258, 262, 268, 270, 252]
n = len(a)
print( '평균', np.mean(a) )

print('그냥 분산', np.var(a))
print('불편 분산', np.var(a, ddof = 1))

print( '불편표준편차 나누기 루트n -->', np.std(a,ddof=1) /np.sqrt(n) )
print( '표준오차 SEM -->', stats.sem(a) )
tu = t.ppf( 0.975, n-1 )
print( 't값', tu )
print( '하한', np.mean(a) - tu*stats.sem(a) )
print( '상한', np.mean(a) + tu*stats.sem(a) )
print( t.interval( 0.95, n-1, loc=np.mean(a), scale=stats.sem(a) ) )
# n - 1 은 자유도 적은거


# %%


# 확인문제 15
from scipy.stats import t

n = 25
x = 35
s = 5

sem = s / np.sqrt(n)

tud = t.ppf(0.975, n - 1) * sem

print(tud)
print("하한: ", x - tud)
print("상한: ", x + tud)


# %%


# 확인문제 16
import numpy as np
from scipy.stats import t


n = 20
x = 170
s = 15

sem = s / np.sqrt(n)

tud1 = t.ppf(0.975, n - 1) * sem
tud2 = t.ppf(0.995, n - 1) * sem

print(tud1)
print("하한: ", x - tud1)
print("상한: ", x + tud1)

print(tud2)
print("하한: ", x - tud2)
print("상한: ", x + tud2)



# %%


# 데이터구조 과제
import random
import math

# 선택 정렬
def selectionSortRec(A, n):                       # 길이 n을 가지는 A 리스트의 선택 정렬(재귀)
  if (n > 2):                                     # n이 2면 원소 한개에 대한 선택 정렬을 하므로 탈출
    k = theLargestRec(A, n - 1)                   # 0에서 n - 1 (n이 0부터 시작하기 때문) 까지에서 최댓값 찾기
    A[k], A[n - 1] = A[n - 1], A[k]               # 최댓값을 맨 마지막으로 보냄

    selectionSortRec(A, n - 1)                    # 재귀

def theLargestRec(A, last:int):                   # 가장 큰 원소 찾는 함수
  largest = 0                                     # 일단 초기값 0으로 설정
  for i in range(last):                           # 0부터 last - 1까지 루프
    if (A[i] > largest + 1):
      largest = i
  return largest                                  # 가장 큰 값 반환


# 버블 정렬
def bubbleSortRec(A, n):
  for i in range(n - 1):                          # 맨 마지막 요소를 빼고 계산 하는 이유가 아래에서 i + 1로 다음 요소까지 찾아버리기 때문
    if (A[i] > A[i + 1]):                         # 오른쪽 값이 더 작으면
      A[i], A[i + 1] = A[i + 1], A[i]             # 서로 바꿈
  if (n > 1):
    bubbleSortRec(A, n - 1)                       # 맨 마지막에 가장 큰 요소가 들어가서 그거 빼고 재귀


# 삽입 정렬
def insertionSortRec(A, start, end):
  value = A[start]                                # 시작하는 위치의 값
  loc = start                                     # 시작하는 위치
  while (loc > 0 and A[loc - 1] > value):         # 위치가 맨 앞이 아니고 이전 위치의 값이 현재 위치의 값보다 작으면 반복
    A[loc] = A[loc - 1]                           # 새로 삽입되는 원소가 정렬된 리스트의 마지막 원소보다 작을 때 왼쪽으로 쉬프트
    loc -= 1                                      # 정렬하는 원소의 위치를 왼쪽으로 이동
  A[loc] = value                                  # 정해진 삽입위치에 삽입하고자 하는 원소 삽입

  if (start + 1 < end):                           # 삽입되는 위치가 가장 마지막이 아닐 때
    insertionSortRec(A, start + 1, end)           # 시작위치를 오른쪽으로 한 칸 옮겨서 재귀


# 병합 정렬
def mergeSort(A, start, end):
    if end - start > 1:  # 배열의 길이가 1보다 크면 정렬을 수행
        mid = (start + end) // 2  # 배열을 반으로 나누기 위한 중간 인덱스
        mergeSort(A, start, mid)  # 왼쪽 반을 재귀적으로 정렬
        mergeSort(A, mid, end)  # 오른쪽 반을 재귀적으로 정렬
        merge(A, start, mid, end)  # 정렬된 두 부분을 병합

def merge(A, start, mid, end):
    left = [A[i] for i in range(start, mid)]  # 왼쪽 부분 배열을 A[start]부터 mid - 1 까지 저장 
    right = [A[i] for i in range(mid, end)]  # 오른쪽 부분 배열을 mid부터 end - 1까지 저장

    i = 0  # 왼쪽 부분의 인덱스
    j = 0  # 오른쪽 부분의 인덱스
    k = start  # 병합된 배열의 인덱스

    while i < len(left) and j < len(right): # i가 왼쪽 부분 배열을 덜 돌았을 때, j가 오른쪽 배열을 덜 돌았을 때
        if left[i] < right[j]:     # 왼쪽의 i번째 수가 오른쪽의 j번째 수보다 작으면
            A[k] = left[i]         # A의 k번째에 저장
            i += 1
        else:
            A[k] = right[j]        # A의 k번째에 저장
            j += 1
        k += 1                     

    while i < len(left):           # 다 돌았는데 왼쪽 요소가 남았을 때
        A[k] = left[i]             # A의 뒤쪽에 다 저장
        i += 1
        k += 1

    while j < len(right):          # 다 돌았는데 오른쪽 요소가 남았을 때
        A[k] = right[j]            # A의 뒤쪽에 다 저장
        j += 1
        k += 1
        
        
# 퀵 정렬
def quickSort(A, p:int, r:int):
    if p < r:
        if r - p + 1 <= 100:  # 배열의 크기가 충분히 작으면(임의로 잡음) 삽입 정렬 사용
            insertionSortRec(A, p, r)
        else:
            q = randomizedPartition(A, p, r)      # 기준 값이 맨 마지막이 아닌 배열 안에 있는 랜덤한 하나의 값으로 기준을 잡음
            quickSort(A, p, q - 1)                # 왼쪽 부분 정렬
            quickSort(A, q + 1, r)                # 오른쪽 부분 정렬

def insertionSortRec(A, start, end):
  value = A[start]                                # 시작하는 위치의 값
  loc = start                                     # 시작하는 위치
  while (loc > 0 and A[loc - 1] > value):         # 위치가 맨 앞이 아니고 이전 위치의 값이 현재 위치의 값보다 작으면 반복
    A[loc] = A[loc - 1]                           # 새로 삽입되는 원소가 정렬된 리스트의 마지막 원소보다 작을 때 왼쪽으로 쉬프트
    loc -= 1                                      # 정렬하는 원소의 위치를 왼쪽으로 이동
  A[loc] = value                                  # 정해진 삽입위치에 삽입하고자 하는 원소 삽입

  if (start + 1 < end):                           # 삽입되는 위치가 가장 마지막이 아닐 때
    insertionSortRec(A, start + 1, end)           # 시작위치를 오른쪽으로 한 칸 옮겨서 재귀

def partition(A, p:int, r:int):                   # 분할
    x = A[r]                                      # 기준 원소
    i = p - 1                                     # 왼쪽 구역의 마지막              이거 집어넣는거 뒤로 집어넣게 하는거 고민해야함
    for j in range(p, r):                         # 오른쪽 구역
        if A[j] < x:                              # j번째 원소가 기준 원소보다 작으면
            i += 1                                # 2구역의 시작점을 오른쪽으로 한칸 이동
            A[i], A[j] = A[j], A[i]               # j번째 원소랑 i번째 원소를 바꾼다
    A[i + 1], A[r] = A[r], A[i + 1]               # 기준 원소와 2구역 첫번째 원소 바꿈
    return i + 1                                  # 기준 원소 위치 리턴

def randomizedPartition(A, p:int, r:int):         # 랜덤으로 시작 원소 결정
    i = random.randint(p, r)                      # 처음과 끝 중에서 위치를 찾음
    A[r], A[i] = A[i], A[r]                       # 해당 위치의 원소를 맨 뒤 원소랑 바꿈
    return partition(A, p, r)                     # 분할


# 힙 정렬
def heapSort(A) :
    buildHeap(A)                                  # 힙 만들기
    for last in range(len(A) - 1, 0, -1) :        # 마지막 원소에 루트노드가 오게 하고 이를 제외한 힙 재생성
        A[last], A[0] = A[0], A[last]             # 마지막 원소와 처음 원소 바꾸기
        percolateDawn(A, 0, last - 1)             # 스며내리기
        
def buildHeap(A) :                                # 힙 만들기
    for i in range((len(A) - 2) // 2, -1, -1) :   # 맨 마지막 노드의 부모노드부터 시작
        percolateDawn(A, i, len(A) - 1)           # 스며내리기
        
def percolateDawn(A, k:int, end:int) :            # 스며내리기
    child = 2 * k + 1                             # 자식 노드 왼쪽
    right = 2 * k + 2                             # 자식 노드 오른쪽
    if (child <= end) :                           # 자식 노드가 마지막 노드가 아닐 때
        if (right <= end and A[child] < A[right]) : # 자식 노드가 마지막 노드가 아니고 오른쪽 자식이 왼쪽 자식보다 클 때
            chile = right                         # 오른쪽 자식 지정
            
        if (A[k] < A[child]) :                    # 지정 노드가 자식 노드보다 작으면
            A[k], A[child] = A[child], A[k]       # 스며내리기 하고
            percolateDawn(A, child, end)          # 다시 본다
        
        
# 쉘 정렬
def shellSort(A) :
    H = gapSequence(len(A))                       # 갭을 얼마나 줄건지 계산
    for h in H :                                  # H에 담긴 갭이 h, 갭의 수 만큼 반복
        for k in range(h) :                       # 0부터 h - 1 까지 반복
            stepInsertionSort(A, k, h)            # 정렬
        
def stepInsertionSort(A, k:int, h:int) :           
    for i in range(k + h, len(A), h) :            # h는 갭이고 k+h부터 시작해 갭만큼 이동, A의 길이만큼 반복
        j = i - h                                 # i에서 갭만큼 왼쪽으로 이동
        newItem = A[i]                            # A 리스트에서 i번째 요소를 저장
        while(0 <= j and newItem < A[j]) :        # j가 0 이상, A[i]가 A[j]보다 작을 때
            A[j + h] = A[j]                       # j + h = i, 오른쪽에 있는 수에 왼쪽의 수를 저장
            j -= h                                # 갭 만큼 왼쪽으로 이동
        A[j + h] = newItem                        # 저장된 요소를 제일 처음 갭 시작 요소에 저장
          
def gapSequence(n:int) :                          # 갭 만들기
    H = [1]; gap = 1                              # 리스트의 시작은 1, 갭도 1
    while (gap < n / 5) :                         # 갭이 A 길이 / 5 보다 작으면
        gap = 3 * gap + 1                         # 지정된 만큼 갭을 정한다
        H.append(gap)                             # 정해진 갭을 리스트에 넣음
    H.reverse()                                   # 리스트를 반대로 돌린다
    return H                                      # 리스트 리턴
        

# 계수 정렬
def countingSort(A) :                             
    k = max(A)                                    # 배열에서 가장 큰 값
    C = [0 for _ in range (k + 1)]                # 해당 값의 크기만큼 배열 생성, 초기화
    
    for j in range(len(A)) :                      # 배열 길이만큼 반복
        C[A[j]] += 1                              # A의 j번째 값이 보이면(예를 들어 2) C[2]의 값에 + 1
    
    for i in range(1, k + 1) :                    # 1부터 k까지 반복
        C[i] += C[i - 1]                          # 각 요소에서 이전 요소 값 더함 - 누적
    
    B = [0 for _ in range(len(A))]                # 새로운 배열 B 초기화
    for j in range(len(A) - 1, -1, -1) :          # A 배열을 뒤에서부터 읽음
        B[C[A[j]] - 1] =  A[j]                    # A[j] 값에 해당하는 C 배열의 요소를 찾아 B에 넣음
        C[A[j]] -= 1                              # C 배열 요소 값 - 1
        
    return B


# 기수 정렬
def radixSort(A) :
    maxValue = max(A)                             # A의 가장 큰 값
    numDigits = math.ceil(math.log10(maxValue))   # 가장 큰 값의 자릿수 계산
    
    bucket = [[] for _ in range(10)]              # 0부터 9까지 빈 리스트

    for i in range(numDigits) :                   # 자릿수만큼 반복
        for x in A :                              # i가 A에 있을 때
            y = (x // 10 ** i) % 10               # i번째 자리에 해당하는 숫자 찾기
            bucket[y].append(x)                   # 각 i번째 자리에 저장
            
        A.clear()                                 # A 지우기
        for j in range(10) :                      
            A.extend(bucket[j])                   # A에 버킷의 j번째 숫자들 추가
            bucket[j].clear()                     # 버킷 지우기
        
        
# 버킷 정렬
def bukitSort(A) :
    n = len(A)                                    # 배열 A의 길이
    B = [[] for _ in range(n)]                    # 길이만큼 빈칸으로 리스트 생성
    for i in range(n) :                           # n만큼 반복
        index = min(n - 1, math.floor(n * A[i]))  # 정수로 변환하여 버킷을 찾음, n - 1을 넘지 않도록 조정
        B[index].append(A[i])                     # A[i] 값을 B에 저장

    A.clear()                                     # A 비움
    for i in range(n) :
        insertionSort(B[i])                       # 버킷 내부 정령
        A.extend(B[i])                            # B를 A로 이동
        
def insertionSort(A) :                            # 삽입정렬
    for i in range(1, len(A)) :                   # 1부터 길이 - 1까지 반복
        loc = i - 1                               # 위치 찾기
        newItem = A[i]                            # 정렬 할 요소 저장
        while (loc >= 0 and newItem < A[loc]) :   # 위치가 0 이상, 해당 요소가 A[loc]보다 작을 때
            A[loc + 1] = A[loc]                   # A[i]에 A[loc] 저장
            loc -= 1                              # loc - 1
        A[loc + 1] = newItem                      # A[i]에 newItem 저장
        
    

# %%

import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt


listLength = 300
sys.setrecursionlimit(listLength * 10000)

sortName = ["선택", "버블", "삽입", "병합", "퀵", "힙", "쉘", "계수", "기수", "버킷"]
times = []

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    selectionSortRec(A, listLength)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('selectionSortRec Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    bubbleSortRec(A, listLength)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('bubbleSortRec Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    insertionSortRec(A, 1, listLength)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('insertionSortRec Time mean: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    mergeSort(A, 0, listLength - 1)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('mergeSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    quickSort(A, 0, listLength - 1)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('quickSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    heapSort(A)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('heapSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    shellSort(A)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('shellSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    countingSort(A)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('countingSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    radixSort(A)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('radixSort Time: ', np.mean(C) * 1000)

C = []
for i in range(10000) :
    A = []
    for value in range(0, listLength):
        A.append(random.randint(0, 100))
    start = time.time()
    bukitSort(A)
    end = time.time()
    C.append(end - start)
times.append(np.mean(C))
print('bukitSort Time: ', np.mean(C) * 1000)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.bar(sortName, times)
plt.xlabel("sort Name")
plt.ylabel("solt time")
plt.title("300개 만번 평균 시간")
plt.show()

# %%

# 데이터구조 누적 정렬시간 그래프










# %%


# 확인문제 15
from scipy.stats import t

n = 25
x = 35
s = 5

sem = s / np.sqrt(n)

tud = t.ppf(0.975, n - 1) * sem

print(tud)
print("하한: ", x - tud)
print("상한: ", x + tud)


# %%


# 확인문제 16
import numpy as np
from scipy.stats import t


n = 20
x = 170
s = 15

sem = s / np.sqrt(n)

tud1 = t.ppf(0.975, n - 1) * sem
tud2 = t.ppf(0.995, n - 1) * sem

print(tud1)
print("하한: ", x - tud1)
print("상한: ", x + tud1)

print(tud2)
print("하한: ", x - tud2)
print("상한: ", x + tud2)


# %%


# 확인문제 19 임시
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 30
trial = 1000

# 데이터 읽기
data = np.loadtxt("C:/파이썬자료/data30000.txt") # 30000개짜리 1차원 배열
print(data[0])

# 1차원 데이터를 1000행 30열 2차원 배열로 바꾸고 싶을 때 바꾸는 라인
data.reshape(trial, n)

print(data[0]) # 0 행에 있는 30개 숫자 출력





# %%


# 확인문제 19
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 30
trial = 1000

# 데이터 읽기
data = np.loadtxt("C:/파이썬자료/data30000.txt")

x = np.zeros(trial) # 1000개의 평균을 저장할 배열 x 초기화
for i in range(trial) :
     x[i] = np.mean(data[i]) # 정규분포

print("표본평균 중 최대 최소: ", np.min(x), np.max(x))
print("표본평균들의 중앙값: ", np.median(x))
print("표본평균들의 평균을 모평균으로 추정: ", np.mean(x))
print("표본평균들의 표준편차 곱하기 루트n: ", np.std(x, ddof = 1) * np.sqrt(n))


y = norm.pdf(x, np.mean(x), np.std(x)) # 확률밀도함수 pdf
plt.bar(x, y)
plt.show()

plt.hist(x, bins = 11, color = 'green') # 히스토그램: 밀도
plt.show()


# %%


# 확인문제 20번
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data = [99.46, 44.22, 99.22, 83.76, 66.62, 42.94, 6.06, 83.29, 31.34, 38.14,
 51.02, 66.29, 22.78, 59.72, 6.93, 80.04, 27.39, 76.67, 85.30, 50.60,
 34.51, 96.39, 9.84, 99.05, 0.16, 27.69, 12.74, 3.52, 7.13, 27.74]

print('모집단 평균 추정: ', np.mean(data))
print('모집단 표준편차 추정: ', np.std(data, ddof = 1)) # 오리지날 데이터기 때문에 곱하는거 없음

plt.hist(data, bins = 6)
plt.show()


# %%


# 확인문제 21번 - 1
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

data = [99.46, 44.22, 99.22, 83.76, 66.62, 42.94, 6.06, 83.29, 31.34, 38.14,
 51.02, 66.29, 22.78, 59.72, 6.93, 80.04, 27.39, 76.67, 85.30, 50.60,
 34.51, 96.39, 9.84, 99.05, 0.16, 27.69, 12.74, 3.52, 7.13, 27.74]

n = 30

zu = norm.ppf(0.975, 0, 1) # 1.96

m = np.mean(data)
s = np.std(data, ddof = 1)

print("하한: ", m - zu * s / np.sqrt(n))
print("상한: ", m + zu * s / np.sqrt(n))

print(norm.interval(0.95, m, s / np.sqrt(n))) # s / np.sqrt(n): 표준 오차
print(t.interval(0.95, n - 1, m, s / np.sqrt(n))) # t 분포는 자유도 추가


# %%


# 확인문제 21번 - 2
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt

data = np.loadtxt("C:/파이썬자료/data30000.txt")

n = 30000

zu = norm.ppf(0.975, 0, 1) # 1.96

m = np.mean(data)
s = np.std(data, ddof = 1)

print("하한: ", m - zu * s / np.sqrt(n))
print("상한: ", m + zu * s / np.sqrt(n))

print(norm.interval(0.95, m, s / np.sqrt(n)))
# 분산이 작으면 효율성, 데이터가 많아지면 일치성


# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_excel("C:/파이썬자료/커피가격.xlsx") # 컬럼명 포함
mu=3055
n=42 # data.size
print(data.describe()) # 참고용

plt.hist(data, label='Coffee Price', bins=7)
plt.legend(); plt.show()


#np.mean(data) == data.mean(), np.std(data, ddof=1) == data.std(ddof = 1)
t = (np.mean(data) - mu) / (np.std(data, ddof=1) / np.sqrt(n))
print('t값 =',t)
# from scipy.stats import t 쓰면 t.cdf
p = 1 - stats.t.cdf(t, n-1) # t값, 자유도 , 1 - P(t < 0.3617)
print('p값 =',p)

result = stats.ttest_1samp(data.coffee, mu, alternative='greater') # 실무용
print(result)

if p<0.05 : print('H0기각')
else : print('H0를 기각할 수 없음')



# %%


# 6장 확인문제 1 차이가 있는지 : 양측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# H0: 점심 평균 비용 mu = 6300원
# H1: 차이가 있다 mu != 6300원

data = pd.read_excel("C:/파이썬자료/S기업_점심비용.xlsx") # 컬럼명 포함
mu=6300
n=100 # data.size
print(data.describe()) # 참고용

plt.hist(data, label='lunch Price', bins=7)
plt.legend(); plt.show()


t = (np.mean(data) - mu) / (np.std(data, ddof=1) / np.sqrt(n))
print('t값 =',t)
p = 1 - stats.t.cdf(t, n-1)
print('p값 =',p)

result = stats.ttest_1samp(data.lunch, mu, alternative='two-sided')
print(result)

if 2 * p < 0.05 : print('H0기각') # 양측이기 떄문
else : print('H0를 기각할 수 없음')



# %%


# 6장 확인문제 2 차이가 있는지 : 양측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# H0: 주당 스마트폰 평균 이용 시간 mu = 35시간
# H1: 차이가 있다 mu != 35시간

data = pd.read_csv("C:/파이썬자료/스마트폰_이용시간.csv", header = None) # 컬럼명 미포함
mu=35
n=40 # data.size
print(data.describe()) # 참고용

plt.hist(data, label='Phone times', bins=7)
plt.legend(); plt.show()


t = (np.mean(data) - mu) / (np.std(data, ddof=1) / np.sqrt(n))
print('t값 =',t)
p = 1 - stats.t.cdf(t, n-1)
print('p값 =',p)

result = stats.ttest_1samp(data, mu, alternative='two-sided')
print(result)

if 2 * p < 0.05 : print('H0기각')
else : print('H0를 기각할 수 없음')



# %%


# 6장 확인문제 3 더 많은 요금: 단측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 가설 H0: 스마트폰 이용 요금 월 평균 mu = 68000
# 가설 H1: 더 많은 요금을 내고 있다 mu > 68000

data = pd.read_csv("C:/파이썬자료/스마트폰_이용요금.txt", header = None) # 컬럼명 미포함
mu=68000
n=80 # data.size
print(data.describe()) # 참고용

plt.hist(data, label='Phone times', bins=8)
plt.legend(); plt.show()


t = (np.mean(data) - mu) / (np.std(data, ddof=1) / np.sqrt(n))
print('t값 =',t)
p = 1 - stats.t.cdf(t, n-1)
print('p값 =',p)

result = stats.ttest_1samp(data, mu, alternative='greater') # 실무용
# greater는 1 - cdf
# less는 그냥 cdf
print(result)

if p < 0.05 : print('H0기각')
else : print('H0를 기각할 수 없음')


# %%


import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
#----------- 길이 다른 두 표본 '분석'할 때는 파일 1개 오케이 --------------#
data = pd.read_excel("C:/파이썬자료/성인_스마트폰_이용시간.xlsx")
print(data.describe())
data.boxplot(column=['male', 'female'], vert=False)
# vert : true면 세로로, false면 가로로
plt.show()
#----------- 길이 다른 두 표본이므로 개별 파일 읽어주세요 ---------------#
d1 = pd.read_excel("C:/파이썬자료/성인_스마트폰_이용시간_남자.xlsx")
d2 = pd.read_excel("C:/파이썬자료/성인_스마트폰_이용시간_여자.xlsx")
result = stats.ttest_ind(d1.male, d2.female, alternative='two-sided')
# 분산이 같음
# p > 0.05 H0 기각 할 수 없다 - H0가 이겼다
# p < 0.05 H0 기각 할 수 있다 - H1가 이겼다
print(result)
# 차이가 있다



# %%


import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
data = pd.read_excel("C:/파이썬자료/대학생_수면시간.xlsx")
print(data.describe())
data.boxplot(column=['male', 'female'], vert=False)
plt.show()

d1 = pd.read_excel("C:/파이썬자료/대학생_수면시간_남자.xlsx")
d2 = pd.read_excel("C:/파이썬자료/대학생_수면시간_여자.xlsx")
result = stats.ttest_ind(d1.male, d2.female, equal_var=False, alternative='two-sided')
# std 차이가 커서 분산이 다르다
# p가 0.05보다 작으므로 H0 기각할 수 있다
print(result)



# %%


import pandas as pd
from scipy import stats
a1 = pd.read_excel("C:/파이썬자료/성인_스마트폰_이용시간_남자.xlsx")
a2 = pd.read_excel("C:/파이썬자료/성인_스마트폰_이용시간_여자.xlsx")
b1 = pd.read_excel("C:/파이썬자료/대학생_수면시간_남자.xlsx")
b2 = pd.read_excel("C:/파이썬자료/대학생_수면시간_여자.xlsx")
result1 = stats.levene(a1.male, a2.female, center='mean')
result2 = stats.levene(b1.male, b2.female, center='mean')
#levene는 var에 true false를 뭘 넣을지 판단하는 함수
print(result1)
print(result2)



# %%


# 예제 5
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
data = pd.read_excel("C:/파이썬자료/피트니스_결과.xlsx")
print(data[['before','after']].describe()) # fm 제외하고 출력
data.boxplot(column=['before', 'after'], vert=False)
plt.show()
result=stats.ttest_rel(data.before, data.after, alternative='greater')
# greater: before > after 위치 중요
# less   : after > before
print(result)

















# %%


# 6장 확인문제 4
import pandas as pd
from scipy import stats
a1 = pd.read_excel("C:/파이썬자료/배터리_지속시간_korea.xlsx")
a2 = pd.read_excel("C:/파이썬자료/배터리_지속시간_china.xlsx")
result1 = stats.levene(a1.korea, a2.china, center='mean')
print(result1)
# p < 0.05 H0 기각, 즉, 이분산. equal_var=False

data = pd.read_excel("C:/파이썬자료/배터리_지속시간.xlsx")
print(data.describe())
data.boxplot(column=['china', 'korea'], vert=False)
plt.show()

d1 = pd.read_excel("C:/파이썬자료/배터리_지속시간_korea.xlsx")
d2 = pd.read_excel("C:/파이썬자료/배터리_지속시간_china.xlsx")
result2 = stats.ttest_ind(d1.korea, d2.china, equal_var=False, alternative='two-sided')
print(result2)
# p 값이 현저하게 작으므로 H0 기각. 즉, 두 공장 배터리 지속시간은 차이가 있다













# %%


# 6장 확인문제 5
import pandas as pd
from scipy import stats
a1 = pd.read_excel("C:/파이썬자료/주당_근로시간.xlsx")
result1 = stats.levene(a1.daejeon, a1.gwangju, center='mean')
print(result1)

data = pd.read_excel("C:/파이썬자료/주당_근로시간.xlsx")
print(data.describe())
data.boxplot(column=['daejeon', 'gwangju'], vert=False)
plt.show()



d1 = pd.read_excel("C:/파이썬자료/주당_근로시간.xlsx")
result2 = stats.ttest_ind(d1.daejeon, d1.gwangju, equal_var=False, alternative='two-sided')
print(result2)
# p 값이 작으므로 H0 기각, 두 지역 근로시간 차이가 있다
















# %%


# 6장 확인문제 6
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
data = pd.read_excel("C:/파이썬자료/확률및통계_성적.xlsx")
print(data[['midterm','final']].describe())
data.boxplot(column=['midterm', 'final'], vert=False)
plt.show()
result = stats.ttest_rel(data.midterm, data.final, alternative='less')
# 부등호 방향에 따라 greater less 달라진다
# >
print(result)
# p 값이 0.05보다 현저하게 작으므로 H0 기각. 즉, 중간보다 기말이 성적 더 높다















# %%


# 6장 확인문제 7
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
data = pd.read_excel("C:/파이썬자료/체중비교.xlsx")
print(data[['before','after']].describe())
data.boxplot(column=['before', 'after'], vert=False)
plt.show()
result = stats.ttest_rel(data.before, data.after, alternative='greater')
# 부등호 방향에 따라 greater less 달라진다
print(result)
# p 값이 0.05보다 현저하게 작으므로 H0 기각. 즉, 체중이 줄었다



# %%



##########################################################################################################
##########################################################################################################
##########################################################################################################
###############          2024 2학년 2학기                                     ############################
###############          2024 2학년 2학기                                     ############################
###############          2024 2학년 2학기                                     ############################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



## 이 밑으로 2025 3학년 1학기 ##




# %%
import pymysql

connect = pymysql.connect(host='127.0.0.1', user='root',
password='1234', db='shop_db', charset='utf8')
cursor = connect.cursor()
sql = "create table usertable \
(userid char(10), username char(10), email char(20), regyear int)"
cursor.execute(sql)
connect.commit() # DDL은 대부분 암시적 커밋
connect.close()
print('테이블 생성 완료!' ) 


# %%

import pymysql

connect = pymysql.connect(host='127.0.0.1', user='root',
password='1234', db='shop_db', charset='utf8')
cursor = connect.cursor()
count = 0
while (True) :
    data1 = input("사용자 ID (Q:종료)==> ")
    if data1 == "q" or data1 == "Q" :
        break;
    data2 = input("사용자 이름 ==> ")
    data3 = input("사용자 이메일 ==> ")
    data4 = input("가입 연도 (정수) ==> ")
    sql = "INSERT INTO usertable VALUES \
        ('" + data1 + "','" + data2 + "','" + data3 + "',"+ data4 + ")"
    cursor.execute(sql)
    count += 1
    print('--------------')
connect.commit() # 데이터가 바뀌는 DML은 커밋 해줘야 실제 저장됩니다.
connect.close()
print(count, ' 건 등록 완료')




# %%

import pymysql
connect = pymysql.connect(host='127.0.0.1', user='root',
password='1234', db='shop_db', charset='utf8')
cursor = connect.cursor()
cursor.execute("SELECT * FROM usertable")
print("사용자ID 사용자이름 이메일 가입연도")
print("---------------------------------------------------------")
while (True) :
    row = cursor.fetchone() # select 해 온 각 행(튜플)
    if row== None :
        break
    print("%-10s %-10s %-20s %d" % (row[0], row[1], row[2], row[3]))
connect.close()







# %%
## 과제 됨

import pymysql
connect = pymysql.connect(host='127.0.0.1', user='root',
password='1234', db='market_db', charset='utf8')
cursor = connect.cursor()
cursor.execute("SELECT * FROM member")
print("사용자ID 사용자이름 이메일 가입연도")
print("---------------------------------------------------------")
while (True) :
    row = cursor.fetchone() # select 해 온 각 행(튜플)
    if row== None :
        break
    print("%-3s %-6s %-2d %-10s %-10s %-10s %d %-10s" % 
          (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
connect.close()




# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












# %%











# %%












