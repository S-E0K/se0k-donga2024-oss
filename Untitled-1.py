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




A
BABA

if(len(S) > len(T)): print(0)
else:
    while(len(S) <= len(T)):
        if(S == T):
            print(1)
            break
        else:
            if(T[-1] == "A"): 
                T = first(T)
            if:T = second(T)
            print(T)
    else: print(0)





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

h = np.loadtxt('C:/Users/김보석/Desktop/대학/2024 - 1학기/확률및통계 01/중학생_남자_키.txt')

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

# 0324 asdf
# %%






# %%







# %%






# %%




