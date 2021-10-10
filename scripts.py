# Say "Hello, World!" With Python

print("Hello, World!")


# Python If-Else

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(raw_input().strip())

if n % 2 != 0:
    print("Weird")
elif n % 2 == 0:
    if n in range(2,6):
        print("Not Weird")
    elif n in range(6, 21):
        print("Weird")
    elif n > 20:
        print("Not Weird")


# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print (a+b)
print (a-b)
print (a*b)


# Python: Division

from __future__ import division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a//b)
print(a/b)


# Loops

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        if i >= 0:
            print(i**2)


# Write a function

def is_leap(year):
    leap = False

    if year % 4 == 0:
        leap = True
        if year % 100 == 0 and year % 400 != 0:
            leap = False
        elif year % 100 == 0:
            leap = True

    return leap


# Print Function

from __future__ import print_function

if __name__ == '__main__':
    n = int(input())

s = ""
for i in range(n):
    s += str(i + 1)

print(s)


# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    results = []
    for i in range(x + 1):
        for j in range(y + 1):
            for k in range(z + 1):
                if i + j + k != n:
                    results.append([i,j,k])
    print(results)


# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

print (sorted(set(arr)) [-2])


# Nested Lists

marksheet = []
scorelist = []
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        marksheet += [[name,score]]
        scorelist += [score]
    b = sorted(list(set(scorelist)))[1]

    for a, c in sorted(marksheet):
        if c == b:
            print(a)


# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line)) #list of scores
        student_marks[name] = scores #key: names, value: list of scores
    query_name = input() #wanted student name
    average = sum(student_marks[query_name])/len(student_marks[query_name])
print("{0:.2f}".format(average))


# Lists

if __name__ == '__main__':
    N = int(input())
    l = []
    for i in range(N):
        x = input().split(' ')
        command = x[0]
        if command == 'insert':
            l.insert(int(x[1]), int(x[2]))
        if command == 'remove':
            l.remove(int(x[1]))
        if command == 'append':
            l.append(int(x[1]))
        if command == 'sort':
            l.sort()
        if command == 'pop':
            l.pop()
        if command == 'reverse':
            l.reverse()
        if command == 'print':
            print(l)


# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
t = tuple(integer_list)
print (hash(t))


# sWAP cASE

def swap_case(s):
    return s.swapcase()


# String Split and Join

def split_and_join(line):
    # write your code here
    line = line.split(" ")
    line = "-".join(line)
    return line


# What's Your Name?

def print_full_name(a, b):
    print("Hello",a,b+"! You just delved into python.")


# Mutations

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = ''.join(l)
    return string


# Find a string

def count_substring(string, sub_string):
    n = 0
    start = 0
    condition = True
    while condition:
        found = string.find(sub_string, start)
        if found != -1:
            n += 1
            start = found + 1
        else:
            condition = False
    return n


# String Validators

if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))


# Text Alignment

thickness = int(input())
c = 'H'

for i in range(thickness):
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

for i in range((thickness+1)/2):
    print (c*thickness*5).center(thickness*6)

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

for i in range(thickness):
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)


# Text Wrap

def wrap(string, max_width):
    return "\n".join([string[i:i+max_width] for i in range(0, len(string), max_width)])


# Designer Door Mat

n, m = map(int,input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))


# String Formatting

def print_formatted(number):
    w = len('{n:b}'.format(n=number))
    for i in range(number):
        print('{0:{w}d} {0:{w}o} {0:{w}X} {0:{w}b}'.format(i+1, w=w))


# Capitalize!

def solve(s):
    for i in s[:].split():
        s = s.replace(i, i.capitalize())
    return s


# Introduction to Sets

def average(array):
    # your code goes here
    s = set(array)
    return sum(s)/len(s)


# No Idea!

n,m = input().split()
el_array = input().split()
A = set(input().split())
B = set(input().split())

happiness = sum((i in A) - (i in B) for i in el_array)
print(happiness)


# Symmetric Difference

m = input()
setM = set(list(map(int, input().split())))
n = input()
setN = set(list(map(int, input().split())))
s = sorted(list(setM.difference(setN))+list(setN.difference(setM)))
for i in s:
    print(i)


# Set .add()

s = set()
for _ in range(int(input())):
    s.add(input())
print(len(s))


# Set .discard(), .remove() & .pop()

xn = int(input())
s = set(map(int, input().split()))

for _ in range(int(input())):
    x = input().split()
    command = x[0]
    if command == 'remove':
        s.remove(int(x[1]))
    if command == 'discard':
        s.discard(int(x[1]))
    if command == 'pop':
        s.pop()
print(sum(s))


# Set .union() Operation

n_english = int(input())
english_set = set(map(int, input().split()))
n_french = int(input())
french_set = set(map(int, input().split()))

print(len(english_set.union(french_set)))


# Set .intersection() Operation

n_english = int(input())
english_set = set(map(int, input().split()))
n_french = int(input())
french_set = set(map(int, input().split()))

print(len(english_set.intersection(french_set)))


# Set .difference() Operation

n_english = int(input())
english_set = set(map(int, input().split()))
n_french = int(input())
french_set = set(map(int, input().split()))

print(len(english_set.difference(french_set)))


# Set .symmetric_difference() Operation

n_english = int(input())
english_set = set(map(int, input().split()))
n_french = int(input())
french_set = set(map(int, input().split()))

print(len(english_set.symmetric_difference(french_set)))

# Set Mutations

N = int(input())
s = set(list(map(int, input().split())))

for _ in range(int(input())):
    x = input().split()
    command = x[0]
    n_set = x[1]

    if command == 'update':
        s.update(set(list(map(int, input().split()))))
    if command == 'intersection_update':
        s.intersection_update(set(list(map(int, input().split()))))
    if command == 'difference_update':
        s.difference_update(set(list(map(int, input().split()))))
    if command == 'symmetric_difference_update':
        s.symmetric_difference_update(set(list(map(int, input().split()))))

print(sum(s))


# The Captain's Room

k = int(input())
listk = list(map(int, input().split()))
setk = set(listk)

print(((sum(setk)*k)-(sum(listk)))//(k-1))


# Check Subset

for _ in range(int(input())):
    A_elements = input()
    setA = set(input().split())
    B_elements = input()
    setB = set(input().split())
    print(setA.issubset(setB))


# Check Strict Superset

A = set(map(int, input().split()))
n_sets = int(input())

strictsuperset = True

for _ in range (n_sets):
    N = set(map(int, input().split()))
    if A.issuperset(N) != True:
        strictsuperset = False

print(strictsuperset)


# collections.Counter()

import collections

n_shoes = int(input())
shoes_sizes = collections.Counter(map(int, input().split()))
n_customers = int(input())

money = 0
for i in range(n_customers):
    x = input().split()
    size = int(x[0])
    xi = int(x[1])
    if shoes_sizes[size]:
        money += xi
        shoes_sizes[size] -= 1

print(money)


# DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)
n, m = list(map(int, input().split()))

for i in range(n):
    d[input()].append(i + 1)

for j in range(m):
    print(' '.join(map(str, d[input()])) or -1)


# Collections.namedtuple()

from collections import namedtuple

n = int(input())
gen = input().split()

marks = 0

for i in range(n):
    x =input().split()
    students = namedtuple('student',gen)
    gen1 = x[0]
    gen2 = x[1]
    gen3 = x[2]
    gen4 = x[3]
    students = students(gen1, gen2, gen3, gen4)
    marks += int(students.MARKS)

print('{:.2f}'.format(marks/n))


# Collections.OrderedDict()

from collections import OrderedDict

n = int(input())
dictionary = OrderedDict()

for _ in range(0,n):
    item, space, value = input().rpartition(" ")
    dictionary[item] = dictionary.get(item,0)+int(value)
for i,j in dictionary.items():
    print (i,j)


# Word Order

distinct_lst = []

for _ in range(int(input())):
    distinct_lst.append(input())

dictionary = {}
for i in distinct_lst:
    if i in dictionary:
        dictionary[i] += 1
    else:
        dictionary[i] = 1
print(len(dictionary))
print(" ".join(str(j) for i,j in dictionary.items()))


# Collections.deque()

from collections import deque
d = deque()

for _ in range(int(input())):
    x = input().split()
    command = x[0]
    if command == 'append':
        d.append(int(x[1]))
    elif command == 'appendleft':
        d.appendleft(int(x[1]))
    elif command == 'pop':
        d.pop()
    elif command == 'popleft':
        d.popleft()

print(*d)


# Company Logo

import math
import os
import random
import re
import sys
import collections


if __name__ == '__main__':
    s = input()

c = collections.Counter(s)
s = sorted(c.items())
c = dict(s)

for k in sorted(c.keys(),reverse=True,key=lambda x:c[x])[:3]:
    print(k,c[k])


# Piling Up!

for _ in range(int(input())):
    n = int(input())
    lst = list(map(int, input().split()))
    l = len(lst)
    i = 0
    while i < l - 1 and lst[i] >= lst[i+1]:
        i += 1
    while i < l - 1 and lst[i] <= lst[i+1]:
        i += 1
    print("Yes" if i == l - 1 else "No")


# Calendar Module

import calendar

ddate = input().split()
m = int(ddate[0])
d = int(ddate[1])
y = int(ddate[2])

wkday = calendar.day_name[calendar.weekday(y,m,d)].upper()
print(wkday)


# Exceptions

n = int(input())

for i in range(n):
    try:
        a,b = map(int,input().split())
        print(a//b)
    except ZeroDivisionError as e:
        print('Error Code:', e)
    except ValueError as v:
        print('Error Code:', v)


# Zipped!

n, x = map(int, input().split())

subjects = []
for i in range(x):
    subjects.append(map(float, input().split()))

students = zip(*subjects)

for student in students:
    sumsub = sum(student)
    average = sumsub / x
    print(average)


# Athlete Sort

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

sorted_arr = sorted(arr, key=lambda record: record[k])

for item in sorted_arr:
    print(*item)


# ginortS

S = sorted(input())

out = []
for s in S:
    if s.islower():
        out.append(s)
for s in S:
    if s.isupper():
        out.append(s)
for s in S:
    if s.isdigit() and int(s) % 2 != 0:
        out.append(s)
for s in S:
    if s.isdigit() and int(s) % 2 == 0:
        out.append(s)

print(''.join(out))


# Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    # return a list of fibonacci numbers
    a,b = 0,1
    for i in range(n):
        yield a
        a,b = b,a+b


# Detect Floating Point Number

import re

n = int(input())

for i in range(n):
    match = re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())
    print(bool(match))


# Re.split()

regex_pattern = r"[.,]+"


# Group(), Groups() & Groupdict()

import re

s = input()
m = re.search(r'([a-zA-Z0-9])\1+', s)
if m:
    print(m.group(1))
else:
    print(-1)


# Re.findall() & Re.finditer()

import re
v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c, v, c), input(), flags = re.I)
print('\n'.join(m or ['-1']))


# Re.start() & Re.end()

import re

lookhere = input()
tofind = input()

m= list(re.finditer("(?=(%s))"%tofind,lookhere))
if not m:
    print((-1,-1))
for i in m:
    print((i.start(1),i.end(1)-1))


# Regex Substitution

import re

n = int(input())

for i in range(n):
    print(re.sub('(?<=\s)\&\&\s', 'and ', re.sub('\s\|\|\s', ' or ', input())))


# Validating Roman Numerals

regex_pattern = r"^M{,3}(C(D|M)|D?C{,3})(X(L|C)|L?X{,3})(I(X|V)|(X|V)?I{,3})$"


# Validating phone numbers

import re

n = int(input())

for i in range(n):
    if re.match(r'[789]\d{9}$', input()):
        print('YES')
    else:
        print('NO')


# Validating and Parsing Email Addresses

import re

n = int(input())

for i in range(n):
    a, b = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', b)
    if m:
        print(a,b)


# Hex Color Code

import re

n = int(input())

for i in range(n):
    for x in re.findall(r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]',input()):
        print(x)


# HTML Parser - Part 1

from html.parser import HTMLParser

n = int(input())


class myHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for elem in attrs:
            print('->', elem[0], '>', elem[1])

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for elem in attrs:
            print('->', elem[0], '>', elem[1])


myParser = myHTMLParser()
for i in range(n):
    myParser.feed(''.join(input().strip()))


# HTML Parser - Part 2

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')

        print(comment)

    def handle_data(self, data):
        if data == '\n': return
        print('>>> Data')
        print(data)


html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

n = int(input())


class myHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print (tag)
        for elem in attrs:
            print ('->',elem[0],'>',elem[1])

myParser = myHTMLParser()
for i in range(n):
    myParser.feed(''.join(input().strip()))


# Validating UID

import re

n = int(input())
p1 = r'[A-Za-z0-9]{10}'
p2 = r'([A-Z].*){2}'
p3 = r'([0-9].*){3}'
p4 = r'.*(.).*\1'

for i in range(n):
    s = input()
    if re.search(p1, s) and re.search(p2, s) and re.search(p3,s) and not re.search(p4,s):
        print('Valid')
    else:
        print('Invalid')


# Validating Credit Card Numbers

import re

n = int(input())

p1 = r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$"
p2 = r"([\d])\1\1\1"

for i in range(n):
    s = input()
    if re.search(p1, s) and not re.search(p2, s.replace("-", "")):
        print('Valid')
    else:
        print('Invalid')


# Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"


# Matrix Script

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

text = [line[i] for i in range(m) for line in matrix]
text = ''.join(text)
print(re.sub(r'([A-Za-z0-9])[!@#$%&\s]+(?=[A-Za-z0-9])',r'\1 ', text))


# XML 1 - Find the Score

def get_attr_number(node):
    count = len(node.attrib)
    for child in node:
        count += get_attr_number(child)
    return count


# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if level == maxdepth:
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)


# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
         f('+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l)
    return fun


# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


# Arrays

def arrays(arr):
    return numpy.array(arr[::-1], float)


# Shape and Reshape

import numpy

llist = input().split()
changeArray = numpy.array(llist, int)
changeArray.shape = (3, 3)
print(changeArray)


# Transpose and Flatten

import numpy

n, m = map(int, input().split())
myArray = numpy.array([input().split() for i in range(n)], int)
print(numpy.transpose(myArray))
print(myArray.flatten())


# Concatenate

import numpy

n, m, p = map(int, input().split())
array1 = numpy.array([input().split() for i in range(n)], int)
array2 = numpy.array([input().split() for i in range(m)], int)

print(numpy.concatenate((array1, array2)))


# Zeros and Ones

import numpy

dims = tuple(map(int, input().split()))

print(numpy.zeros(dims, dtype = numpy.int))
print(numpy.ones(dims, dtype = numpy.int))


# Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

print(numpy.eye(n, m))


# Array Mathematics

import numpy

n, m = map(int, input().split())
a, b = (numpy.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))

print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(numpy.floor_divide(a,b))
print(numpy.mod(a,b))
print(numpy.power(a,b))


# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')

a = numpy.array(input().split(), float)
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))


# Sum and Prod

import numpy

n, m = map(int, input().split())
a = numpy.array([input().split() for i in range(n)], int)

sumA = numpy.sum(a, axis=0)
print(numpy.prod(sumA, axis=0))


# Min and Max

import numpy

n, m = map(int, input().split())
a = numpy.array([input().split() for i in range(n)], int)

mmin = numpy.min(a, axis=1)
print(numpy.max(mmin))


# Mean, Var, and Std

import numpy

n, m = map(int, input().split())
a = numpy.array([input().split() for i in range(n)], float)

print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(round(numpy.std(a), 11))


# Dot and Cross

import numpy

n = int(input())
a = numpy.array([input().split() for i in range(n)], int)
b = numpy.array([input().split() for i in range(n)], int)

print(numpy.dot(a,b))


#Inner and Outer

import numpy

a = numpy.array(tuple(input().split()), int)
b = numpy.array(tuple(input().split()), int)

print(numpy.inner(a,b))
print(numpy.outer(a,b))


# Polynomials

import numpy

a = list(map(float, input().split()))
x = int(input())

print(numpy.polyval(a,x))


# Linear Algebra

import numpy

n = int(input())
a = numpy.array([input().split() for i in range(n)], float)

print(round(numpy.linalg.det(a), 2))


# Birthday Cake Candles

import math
import os
import random
import re
import sys


# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    m = max(ar)
    count = 0
    for i in ar:
        if i == m:
            count += 1
    return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()


# Number Line Jumps

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if v2 < v1 and (x1 - x2) % (v2 - v1) == 0:
        return 'YES'
    else:
        return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# Viral Advertising

import math
import os
import random
import re
import sys


#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    shared = 5
    ppl = 0
    for i in range(n):
        liked = shared // 2
        shared = liked * 3
        ppl = ppl + liked
    return ppl


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    last = arr[n-1]
    flag = False
    for i in range(n-2, -1, -1):
        if arr[i] > last:
            arr[i+1] = arr[i]
            print(*arr, sep=" ", end="\n")
        else:
            arr[i+1] = last
            print(*arr, sep=" ", end="\n")
            flag = True
            break
    if not flag:
        arr[0] = last
        print(*arr, sep=" ", end="\n")



if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2

import math
import os
import random
import re
import sys


# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(1, n):
        for j in range(i):
            if (arr[i] < arr[j]):
                arr[i], arr[j] = arr[j], arr[i]
        print(*arr)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
