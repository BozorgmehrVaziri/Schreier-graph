import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np

alphabet = []
state = []
edge = []
d = int(input("Enter the number of alphabet: "))
for i in range(d):
    letter = input("Enter alphabet letter "+str(i+1)+": ")
    alphabet.append(letter)


print(alphabet)
n = int(input("Enter the number of the automaton states: "))

for i in range(n):
    state.append(input("state" + str(i+1)+": "))

print(state)
edge_lable = []

for j in alphabet:
    for i in state:
        x = input("Action of state " + i + " across the letter " + j + ' :')
        edge.append((i, x))
        edge_lable.append((j, (i,x)))


def t(state, letter):
     for i in range(len(edge_lable)):
        if edge_lable[i][0]==letter and edge_lable[i][1][0]==state:
          return edge_lable[i][1][1]


permutation=[]
for i in state:
   for j in alphabet:

        x = input( i + " ( "+ j + " ) = " )
        permutation.append((i,(j, x)))
print(edge_lable)
print(permutation)

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
l=int(input("Enter the level number: " ))


edge_with_label=[]

if 'e' in state:
    state.remove('e')
for g in state:
    for x in level(l):
        edge.append((x,super_output(g,x)))
        edge_with_label.append((x,super_output(g ,x), g+"|" + super_transition(g,x)))


#print(edge)
print(edge_with_label)

