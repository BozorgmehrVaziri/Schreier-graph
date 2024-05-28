from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import eigvals
import numpy as np
from numpy.linalg import eig
from scipy.sparse import csgraph
alphabet = ['1','2','3','4']
state = ['a',  'b', 'c', 'd', 'e', 'f', 'g']
d = len(alphabet)

edge_lable=[('1', ('a', 'e')), ('1', ('b', 'b')), ('1', ('c', 'c')), ('1', ('d', 'e')), ('1', ('f', 'f')), ('1', ('g', 'e')), ('1', ('e', 'e')), ('2', ('a', 'e')), ('2', ('b', 'e')), ('2', ('c', 'e')), ('2', ('d', 'd')), ('2', ('f', 'f')), ('2', ('g', 'g')), ('2', ('e', 'e')), ('3', ('a', 'a')), ('3', ('b', 'e')), ('3', ('c', 'c')), ('3', ('d', 'e')), ('3', ('f', 'e')), ('3', ('g', 'g')), ('3', ('e', 'e')), ('4', ('a', 'a')), ('4', ('b', 'b')), ('4', ('c', 'e')), ('4', ('d', 'd')), ('4', ('f', 'e')), ('4', ('g', 'e')), ('4', ('e', 'e'))]

permutation=[('a', ('1', '2')), ('a', ('2', '1')), ('a', ('3', '3')), ('a', ('4', '4')), ('b', ('1', '1')), ('b', ('2', '3')), ('b', ('3', '2')), ('b', ('4', '4')), ('c', ('1', '1')), ('c', ('2', '4')), ('c', ('3', '3')), ('c', ('4', '2')), ('d', ('1', '3')), ('d', ('2', '2')), ('d', ('3', '1')), ('d', ('4', '4')), ('f', ('1', '1')), ('f', ('2', '2')), ('f', ('3', '4')), ('f', ('4', '3')), ('g', ('1', '4')), ('g', ('2', '2')), ('g', ('3', '3')), ('g', ('4', '1')), ('e', ('1', '1')), ('e', ('2', '2')), ('e', ('3', '3')), ('e', ('4', '4'))]

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

rep_list1=[x+x+'^-1' for x in state]+[x+'^-1'+x for x in state]+['e']
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
print('Schreier_graph')
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

A = nx.laplacian_matrix(Schreier_graph).todense()
print(A)
# compute the eigenvectors and eigenvalues
eig_vals, eig_vecs = np.linalg.eig(A)
print('eigenvalue')
print(eig_vals)
print('eigenvector')
print(eig_vecs)

# sort the eigenvalues and eigenvectors in descending order
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

print(idx)
print(eig_vals)
print(eig_vecs)
# normalize the eigenvectors
eig_vecs /= np.linalg.norm(eig_vecs, axis=0)

eigvals = np.linalg.eigvals(A)
# Plot the real part of the eigenvalues
plt.hist(np.real(eigvals), bins=500)
#plt.title("Laplacian Spectrum of Schreier graph (level"+str(l)+ ')')
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.show()
plt.savefig("Laplacian Spectrum of Schreier graph (level"+str(l)+ ')',dpi=300)

# assign the eigenvector values to the nodes

Max = eig_vals.argmax()
Min= eig_vals.argmin()

for i, v in enumerate(Schreier_graph.nodes):
    Schreier_graph.nodes[v]['eigenvector'] = eig_vecs[i, Max]

print(nx.get_node_attributes(Schreier_graph, 'eigenvector'))
node=nx.get_node_attributes(Schreier_graph, 'eigenvector')
print(node)


def convert_to_horizontal(string):
    res=''
    for ele in string:
        res += ele + "\n"
    return res

sor=list(Schreier_graph.nodes)
sor.sort()

x=[u for u in sor]
y=[node[u] for u in sor]

plt.plot(x, y,label = "eigenvalue"+str(eig_vals[Max]))

# naming the x axis
plt.xlabel('vertex')
# naming the y axis
plt.ylabel('amplitude')

# giving a title to my graph
#plt.title('The eigenvector is associated with the largest eigenvalue   on level '+str(l))
plt.legend()
# function to show the plot
plt.show()




for g in eig_vals.argsort()[1:3]:
    for i, v in enumerate(Schreier_graph.nodes):
        Schreier_graph.nodes[v]["eigenvector"] = eig_vecs[i, g]
    node = nx.get_node_attributes(Schreier_graph, 'eigenvector')
    x = [u for u in sor]
    y = [node[u] for u in sor]
    plt.plot(x, y, label="eigenvector with eigenvalue : " + str(eig_vals[g]))


plt.xlabel('vertex')
# naming the y axis
plt.ylabel('amplitude')

# giving a title to my graph
#plt.title('The eigenvectors are associated with the first three eigenvalues on the level '+str(l))
plt.legend()
# function to show the plot
plt.show()

