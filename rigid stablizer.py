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
state = ['a','m','b','k','c','e']
d = len(alphabet)

edge_lable=[('0', ('a', 'a')), ('0', ('m', 'm')), ('0', ('b', 'b')), ('0', ('k', 'k')), ('0', ('c', 'e')), ('0', ('e', 'e')), ('1', ('a', 'c')), ('1', ('m', 'a')), ('1', ('b', 'e')), ('1', ('k', 'b')), ('1', ('c', 'e')), ('1', ('e', 'e'))]

permutation=[('a', ('0', '0')), ('a', ('1', '1')), ('m', ('0', '0')), ('m', ('1', '1')), ('b', ('0', '1')), ('b', ('1', '0')), ('k', ('0', '0')), ('k', ('1', '1')), ('c', ('0', '1')), ('c', ('1', '0')), ('e', ('0', '0')), ('e', ('1', '1'))]

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

rep_list1=['aA', 'Aa', 'bB', 'Bb' , 'cC', 'Cc','dD','Dd']+['e']
rep_list2=[]
for x in state:
    u=''
    for i in range(N+1):
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


automaton_edge=[]
if exist_identity==True:
    state.append('e')
for x in state:
    for y in alphabet:
        automaton_edge.append((x, super_transition(x,y),y+'|'+ super_output(x,y)))
print('The automaton: ')
print(automaton_edge)
def vertex_stablizer(vertex,n):
    s=[]
    for j in generate_words(state,n,rep_list3):
        if vertex==super_output(j , vertex):
            s.append(weaker_rep_fun(j))
    return sorted(list(dict.fromkeys(s)),key=len)
v=input('Enter the level of the tree: ')


def list_intersection(lists):
    # Convert each inner list to a set
    sets = [set(lst) for lst in lists]

    # Calculate the intersection of all sets using the `&` operator
    intersection_set = set.intersection(*sets)

    # Convert the intersection set back to a list
    intersection_list = list(intersection_set)

    return intersection_list


def level_stablizer(n):
    return list_intersection([vertex_stablizer(x,N) for x in level(n)])


U=list(dict.fromkeys(list(map(rep_fun,level_stablizer(len(v))))))
U=list(dict.fromkeys(list(map(rep_fun,U))))
print('stablizer of level '+ str(len(v)))

print(sorted(U,key=len))


def rigid_stablizer(vertex):
    U=level(len(vertex))
    U.remove(vertex)
    Y = list(map(weaker_rep_fun,level_stablizer(len(v))))
    Y= list(dict.fromkeys(Y))
    Y.remove('1')
    s=[]
    for g in Y:
        if all(rep_fun(super_transition(g, x))=='1' for x in U):
            s.append(g)
    return sorted(s, key=len)





for x in level(len(v)):
    print('rigid stablizer of vertex '+ x)
    print(sorted(list(dict.fromkeys(list(map(rep_fun, rigid_stablizer(x))))),key=len))


U.remove('1')
def portrate(string, n):
    p=[]
    for x in level(n):
        p.append(rep_fun(super_transition(string,x)))

    if all(super_output(string,x)==x for x in level(n)):
        per='1'
    else:
        per='sigma'
    return (per , p)



