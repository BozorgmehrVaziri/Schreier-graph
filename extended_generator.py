from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import csv
from scipy.linalg import eigvals
import numpy as np
from numpy.linalg import eig
from scipy.sparse import csgraph
alphabet = ['0','1']
state = ['a','b','c','d','A','B','C','D','e']
edge = []
d = len(alphabet)
n = len(state)

edge_lable=[('0', ('a', 'd')), ('0', ('b', 'a')), ('0', ('c', 'a')), ('0', ('d', 'e')), ('0', ('A', 'e')), ('0', ('B', 'A')), ('0', ('C', 'A')), ('0', ('D', 'e')), ('0', ('e', 'e')), ('1', ('a', 'e')), ('1', ('b', 'c')), ('1', ('c', 'a')), ('1', ('d', 'b')), ('1', ('A', 'D')), ('1', ('B', 'C')), ('1', ('C', 'A')), ('1', ('D', 'B')), ('1', ('e', 'e'))]

permutation=[('a', ('0', '1')), ('a', ('1', '0')), ('b', ('0', '0')), ('b', ('1', '1')), ('c', ('0', '0')), ('c', ('1', '1')), ('d', ('0', '0')), ('d', ('1', '1')), ('A', ('0', '1')), ('A', ('1', '0')), ('B', ('0', '0')), ('B', ('1', '1')), ('C', ('0', '0')), ('C', ('1', '1')), ('D', ('0', '0')), ('D', ('1', '1')), ('e', ('0', '0')), ('e', ('1', '1'))]

def t(state, letter):
     for i in range(len(edge_lable)):
        if edge_lable[i][0]==letter and edge_lable[i][1][0]==state:
          return edge_lable[i][1][1]

def o(state,letter):
    for i in range(len(permutation)):
        if permutation[i][0]==state and permutation[i][1][0]==letter:
            return permutation[i][1][1]

def transition(state,word):
    if len(word)<=1:
        return t(state,word)
    else:
        return t(transition(state,word[:-1]),word[-1:])

def output(state,word):
    if len(word)<= 1:
        return o(state,word)
    else:
        return output(state,word[:-1])+o(transition(state,word[:-1]),word[-1:])


def super_output(g,word):
    if len(g)<=1:
        return output(g,word)
    else:
        return super_output(g[:-1],output(g[-1:],word))


def super_transition(g, word):
    if len(g)<=1:
        return transition(g,word)
    else:
        return super_transition(g[:-1],super_output(g[-1:],word))+transition(g[-1:],word)

def convertTuple(v):
    u=''
    for x in v:
        u=u+x
    return u

def level(x):
    s=[seq for seq in itertools.product(alphabet, repeat=x)]
    v= []
    for i in s:
        v.append(convertTuple(i))
    return v

exist_identity=False
if 'e' in state:
    exist_identity=True
    state.remove('e')

N=int(input('Enter the maxmial length of stablier element: '))
l=int(input('Enter the level of the tree: '))
rep_list1=['aA', 'Aa', 'bB', 'Bb' , 'cC', 'Cc','dD','Dd']+['e']
rep_list2=[]
for x in state:
    u=''
    for i in range(N*4+1):
     u = u+x
     rep_list2.append(u)

for x in rep_list2:
    if len(x)==1:
        rep_list2.remove(x)

rep_list2.sort(reverse=True)

rep_list3=['aA', 'Aa', 'bB', 'Bb' , 'cC', 'Cc','dD','Dd']
def rep_fun(s):
    for x in rep_list1:
        s = s.replace(x, '')
    for x in rep_list2:
          s=s.replace(x, x[0]+'^'+str(len(x)))
    if s=='':
        return '1'
    else:
     return s

def weaker_rep_fun(s):
    for x in rep_list1:
        s = s.replace(x, '')
    if s == '':
       return '1'
    else:
       return s

def generate_words(letters, max_length, prohibited_patterns):
    words = []
    for length in range(1, max_length + 1):
        for combination in itertools.product(letters, repeat=length):
            word = ''.join(combination)
            if any(pattern in word for pattern in prohibited_patterns):
                continue
            words.append(word)
    return words

generator=generate_words(state,N,rep_list1)


def iterative_weaker_replace_function(word):
    w=word
    for i in range(0,N*4):
        w=weaker_rep_fun(w)
    return w
def iterative_replace_function(word):
    w=word
    for i in range(0,N*4):
        w=weaker_rep_fun(w)
    return rep_fun(w)

def inverse_fun(x):
    if x==x.upper():
        y=x.lower()
    else:
        y=x.upper()
    return y

def super_inverse_fun(w):
    if len(w)==1:
        return inverse_fun(w)
    else:
        x1=w[0]
        x2=w[1:]
        y=super_inverse_fun(x2)+inverse_fun(x1)
        return y


def Lie_braket(x, y):
    return x+y+super_inverse_fun(x)+super_inverse_fun(y)


def adjoint(x,y):
    return y+x+super_inverse_fun(y)


new_generator=[]

for x in generator:
    for y in generator:
        if not(x==y or x==super_inverse_fun(y)):
            new_generator.append(Lie_braket(x,y))


new_generator= list(dict.fromkeys(list(map(weaker_rep_fun, new_generator))))




def vertex_stablizer(v):
    s=[]
    for j in new_generator:
        if v==super_output(j , v):
            s.append(j)
    return sorted(list(dict.fromkeys(s)),key=len)



def list_intersection(lists):
    # Convert each inner list to a set
    sets = [set(lst) for lst in lists]

    # Calculate the intersection of all sets using the `&` operator
    intersection_set = set.intersection(*sets)

    # Convert the intersection set back to a list
    intersection_list = list(intersection_set)

    return intersection_list

def level_stablizer(l):
    return list_intersection([vertex_stablizer(x) for x in level(l)])



U=list(dict.fromkeys(list(map(iterative_weaker_replace_function,level_stablizer(l)))))
U=list(dict.fromkeys(list(map(rep_fun,U))))
print('stablizer of level '+ str(l))

print(sorted(U,key=len))



def rigid_stablizer(vertex):
    U=level(len(vertex))
    U.remove(vertex)
    Y = list(map(weaker_rep_fun,level_stablizer(l)))
    Y= list(dict.fromkeys(Y))
    Y.remove('1')
    s=[]
    for g in Y:
        if all(iterative_replace_function(super_transition(g, x))=='1' for x in U):
            s.append(g)
    return sorted(s, key=len)




r=[]
for x in level(l):
    print('rigid stablizer of vertex '+ x)
    print(sorted(list(dict.fromkeys(list(map(weaker_rep_fun, rigid_stablizer(x))))),key=len))
    r=r+sorted(list(dict.fromkeys(list(map(iterative_weaker_replace_function, rigid_stablizer(x))))),key=len)

r=list(dict.fromkeys(list(r)))
r.remove('1')
def portrate(string, n):
    p=[]
    for x in level(n):
        p.append(iterative_replace_function(super_transition(string,x)))
    if all(super_output(string,x)==x for x in level(n)):
        per='1'
    else:
        per='sigma'
    return (per , p)
m=int(input('Enter the level of portrate : '))
print('r')
print(r)
for i in r:
    for j in range(1,m):
        print('portrate of element: '+ iterative_replace_function(i) +' on level: '+ str(j))
        print((iterative_replace_function(i), portrate(i,j)))
