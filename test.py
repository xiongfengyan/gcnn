# Some test code.

# 
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

x=np.linspace(-1,1,100) 
y1 = [1 for i in x]
y2 = [i for i in x]
y3 = [2*i*i-1 for i in x]
y4 = [4*i*i*i-3*i for i in x]
y5 = [8*i*i*i*i-8*i*i+1 for i in x]
y6 = [16*i*i*i*i*i-20*i*i*i+5*i for i in x]
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.plot(x,y5)
plt.plot(x,y6)

plt.show()
exit()

A=[[5,-1,-1],[3,1,-1],[4,-2,1]]
lamb, U = np.linalg.eig(A)
print(lamb)
print(U)

exit()
from pylab import *
 
t = arange(0.0, 2.0, 0.01)
s1 = sin(2*pi*t)
s2 = sin(4*pi*t)
 
figure(1)
subplot(211)
plot(t,s1)
subplot(212)
plot(t,2*s1)
 
figure(2)
plot(t,s2)
 
# now switch back to figure 1 and make some changes
figure(1)
subplot(211)
plot(t,s2, 'gs')
setp(gca(), 'xticklabels', [])
 
figure(1)
savefig('fig1')
figure(2)
savefig('fig2')
 
show()

exit()
import sys, os
#sys.path.insert(0, '..')
from lib import models, graph, coarsening, input_data

import tensorflow as tf
import numpy as np
import time
import scipy

sess = tf.InteractiveSession()
labels = [0,0,1,1,0,1,0,2,1,2,0,3]
labelset = set(labels)
print(labelset)
'''
t = tf.constant([[[ 1, 1], [ 2, 2], [ 3, 3], [ 4, 4], [ 5, 5]],
                 [[ 6, 6], [ 7, 7], [ 8, 8], [ 9, 9], [10,10]],
                 [[11,11], [12,12], [13,13], [14,14], [15,15]],
                 [[16,16], [17,17], [18,18], [19,19], [20,20]],
                 [[21,21], [22,22], [23,23], [24,24], [25,25]]], dtype=tf.float32)
f = tf.constant([[[1,1], [1,1], [1,1]],
                 [[1,1], [2,2], [1,1]],
                 [[1,1], [1,1], [1,1]]], dtype=tf.float32)

t=tf.expand_dims(t, 0)
#f=tf.expand_dims(f, 2)
f=tf.expand_dims(f, 3)
print(sess.run(tf.shape(t)))
print(sess.run(tf.shape(f)))
op1 = tf.nn.conv2d(t, f, strides=[1, 1, 1, 1], padding='VALID')
print(sess.run(op1))
'''
'''
a=[[1.0, 3.0, 5.0, 7.0],
   [8.0, 6.0, 4.0, 2.0],
   [4.0, 2.0, 8.0, 6.0],
   [1.0, 3.0, 5.0, 7.0]]
L = scipy.sparse.csr_matrix(a)
Lmax = graph.lmax(L, normalized=False)
print(Lmax)
L = graph.rescale_L(L, lmax=Lmax)
print(L)
L = L.tocoo()
indices = np.column_stack((L.row, L.col))
L = tf.SparseTensor(indices, L.data, L.shape)
L = tf.sparse_reorder(L)
#pooling=tf.nn.max_pool(a,[1,2,2,1],[1,2,2,1],padding='VALID')  
with tf.Session() as sess:
    #print("image:")
    #image=sess.run(a)
    #print(image) 
    #print("reslut:")
    print(L)
'''
'''
def grid_graph(m, corners=False):
    z = graph.grid(m)
    k=3
    #print(z)
    #print(np.array(z).shape)
    dist, idx = graph.distance_sklearn_metrics(z, k=k, metric='euclidean')
    #print(np.array(dist).shape)
    #print(np.array(idx).shape)
    #print(dist)
    #print(idx)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, k*m**2//2))
    return A    
t_start = time.process_time()
coarsening_levels=1
A = grid_graph(3, corners=False)
print('Step1: ================')
print(A)
graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
#print(len(perm))
for A in graphs:
    print('--------------------------')
    print(A.shape)
    print(A)
print(perm)
#L = [graph.laplacian(A, normalized=True) for A in graphs]
#graph.plot_spectrum(L)
#L = [graph.laplacian(A, normalized=True) for A in graphs]

train=np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])
print(train.shape)
print(train)
print('--------------------------------')
train = coarsening.perm_data(train, perm)
print(train.shape)
print(train)

print('-++++++++++++++++++++++++++++++-')
import scipy
#W1=scipy.sparse.csr_matrix([[5,-1,-1],[3,1,-1],[4,-2,1]],shape=(3,3))
W2=np.array([[5,-1,-1],[3,1,-1],[4,-2,1]])
W1=np.array([[1,1,1],[1,1,1],[1,2,2]])
#print(np.mat(W1).I)
'''
W2=np.array([[1.3,-0.8,-0.5,0,0],
             [-0.8,1.5,-0.7,0,0],
             [-0.5,-0.7,1.2,0,0],
             [0,0,0,0.6,-0.6],
             [0,0,0,-0.6,0.6]])
'''
#from numpy import linalg as LA
#w, v = np.linalg.eig(np.diag((1, 2, 3)))  
#print(w)
#print(v)
a,b=np.linalg.eig(W2)
print(np.diag(a))
print(np.mat(b).I)
print('----WW--------')
WW=np.dot(np.dot(b,np.diag(a)),np.mat(b).I)
print(WW)
for i in [3,2,2]:
    det=(5-i)*(1-i)*(1-i)+4+6+4*(1-i)-2*(5-i)+3*(1-i)
    print(det)
#lamb, U=graph.fourier(W1,k=2)
print(len(a))
print(a[0],b[:,0])
for i in range(0,len(a)):
    print('+++++-----------+++++')
    #print(b[i].T.shape)
    det=np.dot(W2,b[:,i])
    pet=a[i]*b[:,i]
    print(det-pet)
    #print(det)
    #print(pet)
print(a)
print(b)
import sklearn.manifold
graph=scipy.sparse.csr_matrix(W2)
print(graph)
#spectral_embedding=sklearn.manifold.spectral_embedding(graph, 3)
#L=graph.laplacian(W1,False)
#print(L)
'''