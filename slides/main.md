name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Python Numpy tutorial
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/python-numpy]
### .x-small[https://yj-yu.github.io/python-numpy]

.bottom.img-66[ ![](images/lablogo.png) ]

---
layout: false

## About
- Python Tutorial(wrap up) and numpy tutorial
- Numpy, scipy tutorials and enhanced references
- Python practice

---


template: inverse

# Before the practice..


---

## Be pythonic

The Official Python Tutorial
- https://docs.python.org/2/tutorial/

Wikibooks’ Non-Programmers Tutorial for Python
- https://en.wikibooks.org/wiki/Non-Programmer%27s_Tutorial_for_Python_3/Intro

Build dynamic web site (Nettuts+'s python, Django Book)
- https://code.tutsplus.com/series/python-from-scratch--net-20566
- http://www.djangobook.com/en/2.0/index.html

---
## Be pythonic

Make game with python (Invent with Python, Build a Python Bot)
- http://inventwithpython.com/chapters/
- https://code.tutsplus.com/tutorials/how-to-build-a-python-bot-that-can-play-web-games--active-11117

If you want to learn Computer Science, Algorithm from python
- Think Python: How to Think Like a Computer Scientist) http://www.greenteapress.com/thinkpython/thinkpython.html
- 번역글 : http://www.flowdas.com/thinkpython/

시간 있으실 때 무료 e-book 및 tutorial을 해보시기 바랍니다.
다른거 힘들게 배울 필요 없습니다.


---
## Be pythonic

Pycon KR에 참가도 해보세요
프로그래밍으로 상상한 거의 모든것이 가능합니다.

- https://www.pycon.kr/2017/program/schedule/


---

template: inverse

# Python Numpy tutorial


---

## Numpy & Scipy
대표적인 수식 라이브러리, 서로 친합니다.
- 보통 SciPy가 NumPy에 의존합니다. 
- 수학적이거나 과학적인 연구를 위한 진지한 계산 처리를 한다면 이 두 라이브러리로 충분

NumPy와 SciPy는 
- 파이썬의 수학적인 함수와 능력을 확장해주고 
- 작업들을 엄청나게 가속시켜 줍니다

---


## Install configuration

```python
git clone --recursive https://github.com/yj-yu/python-numpy.git
cd python-numpy
ls code
```

code(https://github.com/yj-yu/python-numpy)

```bash
./code
├── numpy-tutorial (from rougier)
├── scipy-2016-sklearn (from amueller)
└── cs228-materials

```

- numpy-tutorial : https://github.com/rougier/numpy-tutorial
- scipy-2016-sklearn : https://github.com/amueller/scipy-2016-sklearn
- cs228-materials : https://github.com/kuleshov/cs228-material.git
  한글 버전 (AI korea) http://aikorea.org/cs231n/python-numpy-tutorial/
---
## 만약 git clone이 오래걸릴 경우

numpy-tutorial과 scipy는 자료가 많습니다. 
따라서 오래 걸리시는분의 경우, 홈페이지는 --recursive로 받지 않으셔도 됩니다.

```bash
git clone https://github.com/yj-yu/python-numpy.git
# 필요한 자료만
git clone https://github.com/rougier/numpy-tutorial.git
git clone https://github.com/kuleshov/cs228-material.git 
git clone https://github.com/amueller/scipy-2016-sklearn.git 
```

선별적으로 받아주세요
이번 시간에 같이 실습하지 않고, 연습을 위해 첨부한 자료 numpy-tutorial , scipy-2016-sklearn 두개는 지금 당장 받지 않으셔도 됩니다.

---
```python
cd code
cd cs228-material/tutorials/python/
jupyter notebook
```

Let's start!
---
## Python Tutorial
Adapted by Volodymyr Kuleshov and Isaac Caswell from the CS231n Python tutorial by Justin Johnson (http://cs231n.github.io/python-numpy-tutorial/).

### Introduction
Python is a great general-purpose programming language on its own, 
but with the help of a few popular libraries (numpy, scipy, matplotlib) 
it becomes a powerful environment for scientific computing.

---
## Python Tutorial

In this tutorial, we will cover:
Basic Python: Basic data types 
- Containers, Lists, Dictionaries, Sets, Tuples, Functions, Classes
- Numpy: Arrays, Array indexing, Datatypes, Array math, Broadcasting
- Matplotlib: Plotting, Subplots, Images
- IPython: Creating notebooks, Typical workflows


---
## Python (review)

Python is a high-level, dynamically typed 

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
```

---
## Basic Data Types

### Numbers
Integers and floats work as you would expect from other languages:

```python
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

---
## Basic Data Types

### Booleans
Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):

```python
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
```

---
## Basic Data Types

### Strings 
Python has great support for strings:

```python
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

---
## Basic Data Types

### Strings 

String objects have a bunch of useful methods; for example:

```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

---
## Containers

Python includes several built-in container types: lists, dictionaries, sets, and tuples.

### Lists

A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:
```python
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
As usual, you can find all the gory details about lists in the documentation.
```
---
## Containers

###Slicing
In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:
```python
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```
We will see slicing again in the context of numpy arrays.

---
## Containers

### Loops
You can loop over the elements of a list like this:
```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
```

If you want access to the index of each element within the body of a loop, use the built-in enumerate function:
```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```
---
## Containers

### Loops

When programming, frequently we want to transform one type of data into another. 
As a simple example, consider the following code that computes square numbers:
```python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]
```
You can make this code simpler using a list comprehension:
```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]
```

List comprehensions can also contain conditions:
```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
```

---
## Containers

### Dictionaries
A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript. You can use it like this:
```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```
You can find all you need to know about dictionaries in the documentation.
---
## Containers

### Dictionaries

Loops: It is easy to iterate over the keys in a dictionary:
```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```
---
## Containers

### Dictionaries

If you want access to keys and their corresponding values, use the items method:
```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

---
## Containers

### Dictionaries
Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:
```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```
---
## Containers

### Sets
A set is an unordered collection of distinct elements. 
As a simple example, consider the following:
```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```

---
## Containers

### Sets
Loops: Iterating over a set has the same syntax as iterating over a list.
however sets are unordered

```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```

---
## Containers 
### Set comprehensions
Like lists and dictionaries, we can easily construct sets using set comprehensions:
```python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
```
---
## Containers

### Tuples

A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

---
## Functions
Python functions are defined using the def keyword. For example:

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```
---
## Functions

We will often define functions to take optional keyword arguments, like this:
```python
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```

---
## Classes

The syntax for defining classes in Python is straightforward:
```python
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

---
## Numpy
Numpy is the core library for scientific computing in Python.

### Arrays
A numpy array is a grid of values, all of the same type

```python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
```
---
## Numpy

### Arrays
Numpy also provides many functions to create arrays:

```python
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

---
## Numpy

### Array indexing
Numpy offers several ways to index into arrays.

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
```

See several indexing examples in tutorials!

---
## Numpy : Datatypes
Every numpy array is a grid of elements of the same type.

```python
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```

---
## Numpy : Array math
Basic mathematical functions operate elementwise on arrays, 
and are available both as operator overloads and as functions in the numpy module:

```python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))
```
---
## Numpy : Array math

```python
# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```
---
## Numpy : Array math

```python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

---
## Numpy : Array math

Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

```python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

---
## Numpy : Array math

Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:

```python
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

---
## Numpy : Broadcasting
Boadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes

For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```
This works; however when the matrix x is very large, computing an explicit loop in Python could be slow.

---
## Numpy : Broadcasting

We could implement this approach like this:
```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```
---
## Numpy : Broadcasting

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```
---
## Numpy : Broadcasting

Play another tutorial blocks 

---
## Scipy
SciPy builds on Numpy, 
provides a large number of functions that operate on numpy arrays 
useful for different types of scientific and engineering applications.

### You can practice scipy with scipy-2016-sklearn

---
## Scipy : Image operation

it has functions to read images from disk into numpy arrays, 
to write numpy arrays to disk as images, and to resize images. 

```python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```
.bottom.center.img-33[ ![](images/cat.png) ]

---


name: last-page
class: center, middle, no-number
## Thank You! See you next week!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: 변훈 연구원님, 송재준 교수님</p>
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
