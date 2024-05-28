from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import eigvals
import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm



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
        x = input("Choose the target state for state " + i + " through letter  " + j + ": ")
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
l=int(input('Enter the dimension of alphabet: '))
edge=[]
edge_with_label=[]
exist_identity=False

if 'e' in state:
    exist_identity = True
    state.remove('e')
for g in state:
    for x in level(l):
        edge.append((x,super_output(g,x)))
        edge_with_label.append((x,super_output(g ,x), g+"|" + super_transition(g,x)))


print(edge)
print(edge_with_label)

Schreier_graph=nx.Graph(edge)

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

# define your graph and its vertices

vertices = list(Schreier_graph.nodes())
Lap=nx.adjacency_matrix(Schreier_graph).todense()

# assign the eigenvector values to the nodes
l=[]
for v in vertices:
    l.append((v, eig_vecs[vertices.index(v), 0]))

print(l)



#first_point=input('choose a first point:')
first_point=vertices[1]
#rate=int(input('enter the rate of '+first_point+':'))
rate=1

initial_step =[]
for x in vertices:
    if x!= first_point:
       initial_step.append(0)
    else:
        initial_step.append(rate)

print(type(vertices))
step_number=int(input("enter th number of steps: "))

diffusion_header=['Id']
for i in range(1,step_number+1):
    diffusion_header.append('step'+str(i))

res= [initial_step]
for s in range(1,step_number):
    v=np.dot(Lap,res[s-1]).tolist()
    res.append((v/norm(np.dot(Lap,res[s-1]))).tolist())
print(np.transpose(res))

with open('Diffusion frames.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(diffusion_header)

    for x in vertices:
        t=list(np.transpose(res)[vertices.index(x)])
        print([x]+t)
        writer.writerow([x]+t)
file.close()


