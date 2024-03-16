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

print(a.sum())
print(a.mean())
print(a.var())
print(a.std())
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





# %%







# %%







# %%






