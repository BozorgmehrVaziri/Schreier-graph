from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import eigvals
import numpy as np
from numpy.linalg import eig
from scipy.sparse import csgraph
alphabet = ['0','1']
state = ['a','b','c','d','A', 'B','C','D','e']
d = len(alphabet)

edge_lable=[('0', ('a', 'd')), ('0', ('b', 'a')), ('0', ('c', 'a')), ('0', ('d', 'e')), ('0', ('A', 'e')), ('0', ('B', 'A')), ('0', ('C', 'A')), ('0', ('D', 'e')), ('0', ('e', 'e')), ('1', ('a', 'e')), ('1', ('b', 'c')), ('1', ('c', 'a')), ('1', ('d', 'b')), ('1', ('A', 'D')), ('1', ('B', 'C')), ('1', ('C', 'A')), ('1', ('D', 'B')), ('1', ('e', 'e'))]

permutation=[('a', ('0', '1')), ('a', ('1', '0')), ('b', ('0', '0')), ('b', ('1', '1')), ('c', ('0', '0')), ('c', ('1', '1')), ('d', ('0', '0')), ('d', ('1', '1')), ('A', ('0', '1')), ('A', ('1', '0')), ('B', ('0', '0')), ('B', ('1', '1')), ('C', ('0', '0')), ('C', ('1', '1')), ('D', ('0', '0')), ('D', ('1', '1')), ('e', ('0', '0')), ('e', ('1', '1'))]
edge=[]
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
l=int(input('Enter the dimension of alphabet: '))

edge_with_label=[]

exist_identity=False
if 'e' in state:
    exist_identity=True
    state.remove('e')

for g in state:
    for x in level(l):
        edge.append((x,super_output(g,x)))
        edge_with_label.append((x,super_output(g,x), g+"|"+ super_transition(g,x)))

import csv

header = ['source', 'target','label']
with open('Schreier graph.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(header)

    for x in edge_with_label:
        writer.writerow(x)
file.close()

n=int(input('Enter the dimension of automaton generator: '))

rep_list1=['aA', 'Aa', 'bB', 'Bb' , 'cC', 'Cc','dD','Dd']+['e']
rep_list2=[]
for x in state:
    u=''
    for i in range(n+1):
     u = u+x
     rep_list2.append(u)

for x in rep_list2:
    if len(x)==1:
        rep_list2.remove(x)

rep_list2.sort(reverse=True)
print(rep_list2)
def rep_fun(s):
    for x in rep_list1:
        s = s.replace(x, '')
    for x in rep_list2:
          s=s.replace(x, x[0]+'^'+str(len(x)))
    if s=='':
        return '1'
    else:
     return s


def gen(x):
    s=[seq for seq in itertools.product(state, repeat=x)]
    v= []
    for i in s:
        v.append(convertTuple(i))
    return v
generator=[]
for i in range(n):
    generator=generator+gen(i+1)
print(generator)
Schreier_graph=[]
for i in level(l):
    for j in generator:
        Schreier_graph.append((i, super_output(j,i), rep_fun(j)+ ' | ' + rep_fun(super_transition(j,i))))

print(Schreier_graph)

print(generator)
Moor_graph=[]
for letter in level(l):
    for g in generator:
        Moor_graph.append((rep_fun(g),rep_fun(super_transition(g,letter)), letter+' | '+super_output(g,letter)))

print(Moor_graph)

Schreier_header = ['source', 'target','label']
with open('Schreier graph of rank '+str(n)+' on level' +str(l)+'.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(Schreier_header)

    for x in Schreier_graph:
        writer.writerow(x)
file.close()

Moor_header = ['source', 'target','label']
with open('Moor graph of rank '+str(n)+' on level' +str(l)+'.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(Moor_header)

    for x in Moor_graph:
        writer.writerow(x)
file.close()

automaton_edge=[]
if exist_identity==True:
    state.append('e')
for x in state:
    for y in alphabet:
        automaton_edge.append((x, super_transition(x,y),y+'|'+ super_output(x,y)))
print('The automaton: ')
print(automaton_edge)

Moor_header = ['source', 'target','label']
with open('Automaton.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(Moor_header)

    for x in automaton_edge:
        writer.writerow(x)
file.close()

Schreier_graph=nx.Graph(edge)
print(nx.is_connected(Schreier_graph))
diam=nx.diameter(Schreier_graph)
distance=nx.resistance_distance(Schreier_graph,level(l)[0],level(l)[-1])
print(diam)
print(distance)

# Compute the Laplacian matrix
L = nx.laplacian_matrix(Schreier_graph).todense()

# Compute the eigenvalues of the Laplacian
eigvals = np.linalg.eigvals(L)
print(eigvals)



# Plot the real part of the eigenvalues
plt.hist(np.real(eigvals), bins=500)
plt.title("Laplacian Spectrum of Schreier graph (level"+str(l)+ ')')
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.show()
plt.savefig("Laplacian Spectrum of Schreier graph (level"+str(l)+ ')',dpi=300)

A = nx.to_numpy_array(Schreier_graph)
print(A)
# compute the eigenvectors and eigenvalues
eig_vals, eig_vecs = np.linalg.eig(A)

# sort the eigenvalues and eigenvectors in descending order
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# normalize the eigenvectors
eig_vecs /= np.linalg.norm(eig_vecs, axis=0)

# assign the eigenvector values to the nodes
for i, v in enumerate(Schreier_graph.nodes):
    Schreier_graph.nodes[v]['eigenvector'] = eig_vecs[i, 0]

print(nx.get_node_attributes(Schreier_graph, 'eigenvector'))
