

```python
# 최성철 파이썬 입문 8장
# 여러단어들을 하나로 붙일 때

colors = ['red', 'blue', 'green', 'yellow']
result = ','.join(colors)
print(result)
```

    red,blue,green,yellow
    


```python
# Split 함수

# String Type의 값을 나눠서 List 형태로 변환

items = 'zero one two three'.split()
print(items)
```

    ['zero', 'one', 'two', 'three']
    


```python
# 리스트에 있는 각 값을 a,b,c 변수로 unpacking
example = 'python,jquery,javascript'
example.split(",")
a,b,c = example.split(",")
print(a,b,c)
```

    python jquery javascript
    


```python
# 리스트 활용

result = [i for i in range(10)]
print(result)

result = [i for i in range(10) if i % 2 == 0]
print(result)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [0, 2, 4, 6, 8]
    


```python
# 스트링 합치기
word1 = "Hello"
word2 = "World"
result = [i + j for i in word1 for j in word2]
print(result)

case1 = ['A','B','C']
case2 = ['D','E','F']
result = [i + j for i in case1 for j in case2 if not(i==j)]
print(result)
```

    ['HW', 'Ho', 'Hr', 'Hl', 'Hd', 'eW', 'eo', 'er', 'el', 'ed', 'lW', 'lo', 'lr', 'll', 'ld', 'lW', 'lo', 'lr', 'll', 'ld', 'oW', 'oo', 'or', 'ol', 'od']
    ['AD', 'AE', 'AF', 'BD', 'BE', 'BF', 'CD', 'CE', 'CF']
    


```python
words = 'The quick brown fox jumps over the lazy dog'.split()
print(words)
stuff = ([w.upper(), w.lower(), len(w)] for w in words)
for i in stuff:
    print(i)
```

    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    ['THE', 'the', 3]
    ['QUICK', 'quick', 5]
    ['BROWN', 'brown', 5]
    ['FOX', 'fox', 3]
    ['JUMPS', 'jumps', 5]
    ['OVER', 'over', 4]
    ['THE', 'the', 3]
    ['LAZY', 'lazy', 4]
    ['DOG', 'dog', 3]
    


```python
# Two dimensional vs One dimensional
case1 = ["A","B","C"]
case2 = ["D","E","A"]
result = [i+j for i in case1 for j in case2]
print(result)
result = [[i+j for i in case1] for j in case2]
print(result)
```

    ['AD', 'AE', 'AA', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA']
    [['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'], ['AA', 'BA', 'CA']]
    


```python
# Enumerate 
for i, v in enumerate(['tic','tac','toe']):
    print(i,v)

mylist = ["a","b","c","d"]
print(list(enumerate(mylist)))

# 문장을 list로 만들고 list의 index와 값을 unpacking하여 dict로 저장

{i:j for i,j in enumerate('KEI Bigdata Research Team'.split())}
```

    0 tic
    1 tac
    2 toe
    [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]
    




    {0: 'KEI', 1: 'Bigdata', 2: 'Research', 3: 'Team'}




```python
# Zip
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for a, b in zip(alist, blist):
    print(a,b)

a,b,c =zip((1,2,3),(10,20,30),(100,200,300))
sum_list = [sum(x) for x in zip((1,2,3), (10,20,30), (100,200,300))]
print(sum_list)
```

    a1 b1
    a2 b2
    a3 b3
    [111, 222, 333]
    


```python
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for i, (a, b) in enumerate(zip(alist, blist)):
    print (i, a, b) # index alist[index] blist[index] 표시
```

    0 a1 b1
    1 a2 b2
    2 a3 b3
    
