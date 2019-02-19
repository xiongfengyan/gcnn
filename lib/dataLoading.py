# Copyright 2018 Wuhan Univeristy. All Rights Reserved.
# 2018-01-22
#
# =====================================================#
# ==============================================================================
"""Functions for loading buildings data."""

import json, os, datetime
import numpy as np
from lib import graph

import scipy, sklearn, scipy.sparse.csgraph
from sklearn import metrics
from scipy.spatial import Delaunay

def load_data(filename, dataSeparation = None):
    print("Loading the {} data".format(filename))
    file = open(filename,'r',encoding='utf-8')
    data = json.load(file)
    file.close()
    if (dataSeparation != None):
        if len(dataSeparation) == 3:
            return separating_dataset_three(data, dataSeparation)
    else:
        return constructingGraph(data, -1)[:6]

def separating_dataset_three(inFIDDic, dataSeparation):
    # Separating dataset to train, validate and test.
    train_count,val_count,test_count = round(len(inFIDDic)*dataSeparation[0]),\
                                       round(len(inFIDDic)*dataSeparation[1]),\
                                       len(inFIDDic)-round(len(inFIDDic)*dataSeparation[0])-round(len(inFIDDic)*dataSeparation[1])
    labelsDic,classes={},[]
    for k in inFIDDic:
        label=inFIDDic[k][0]
        if labelsDic.get(label)==None: 
            labelsDic[label] = 1
        else:
            labelsDic[label] += 1
    #assert len(labelsDic)==2

    for k in labelsDic:
        oneClass = [k, 
                    round(labelsDic[k] * dataSeparation[0]), 
                    round(labelsDic[k] * dataSeparation[1]),
                    int(labelsDic[k]) - round(labelsDic[k] * dataSeparation[0]) - round(labelsDic[k] * dataSeparation[1])]
        classes.append(oneClass)
    classes=np.array(classes)

    print("Train datasets:")
    print(" classes train_num val_num test_num total_num")

    for i in range(0,len(classes)):
        print("{0: ^8}{1: ^10}{2: ^9}{3: ^9}{4: ^9}".format(classes[i][0],classes[i][1],classes[i][2],classes[i][3],sum(classes[i,1:])))
    print("{0: ^8}{1: ^10}{2: ^9}{3: ^9}{4: ^9}".format(" total",sum(classes[:,1]),sum(classes[:,2]),sum(classes[:,3]),classes[:,1:].sum()))

    train_Dic, val_Dic, test_Dic={}, {}, {}
    for k in inFIDDic:
        label=inFIDDic[k][0]
        index=np.argwhere(classes[:,0]==label)[0][0].astype(np.int64)
        if (classes[index][1] > 0):
            train_Dic[k]=inFIDDic[k]
            classes[index][1] = classes[index][1]-1
        elif (classes[index][2] > 0):
            val_Dic[k]=inFIDDic[k]
            classes[index][2] = classes[index][2]-1
        else:
            test_Dic[k]=inFIDDic[k]
            classes[index][3] = classes[index][3]-1

    train_data=constructingGraph(train_Dic, 1)

    return train_data[:5],\
           constructingGraph(val_Dic, 2,train_data[5],train_data[6])[:5],\
           constructingGraph(test_Dic,3,train_data[5],train_data[6])[:5]

def constructingGraph(inFIDDic, data_type,
                      mean_geometry=0, std_geometry=1, is_distance=True):
    #inFiDDic {key,[label,pointlist]}
    if len(inFIDDic) < 1: return None

    vertices_Geometry, adjacencies, labels, inFIDs, process_count = [], [], [], [], 0

    for k in inFIDDic:
        [label, Node_coords, Node_features] = inFIDDic[k]
        assert len(Node_coords) == len(Node_features)
        subObject_size = len(Node_coords)

        # # 1 get the label of this sample.
        label=1 if label == 3 else 0

        # # 3 get the adjacency graph of the building group (one sample).
        # #   MST, Delaunay, K-NN
        points = np.array(Node_coords)
        adjacency = np.zeros((subObject_size, subObject_size))

        tri = Delaunay(points[:,0:2])

        for i in range(0, tri.nsimplex):
            if i > tri.neighbors[i,2]:
                adjacency[tri.vertices[i,0], tri.vertices[i,1]] = 1
                adjacency[tri.vertices[i,1], tri.vertices[i,0]] = 1
            if i > tri.neighbors[i,0]:
                adjacency[tri.vertices[i,1], tri.vertices[i,2]] = 1
                adjacency[tri.vertices[i,2], tri.vertices[i,1]] = 1
            if i > tri.neighbors[i,1]:
                adjacency[tri.vertices[i,2], tri.vertices[i,0]] = 1
                adjacency[tri.vertices[i,0], tri.vertices[i,2]] = 1

        adjacency = scipy.sparse.coo_matrix(adjacency, shape=(subObject_size, subObject_size))
        # In order to make the calculation simpler, only the distance between the center points of the buildings is provided here. 
        # According to the author's experience, the closest distance of two building outlines coule be a better opition for this task.
        distances = sklearn.metrics.pairwise.pairwise_distances(points[:,0:2], metric="euclidean", n_jobs=1)

        if False:
            # K-nearest neighbor graph.
            # Distance matrix. is it necessary to be normalized?    
            idx = np.argsort(distances)[:, 1:1+1]
            distances.sort()
            distances = graph.adjacency(distances[:, 1:1+1], idx)
            adjacency = scipy.sparse.coo_matrix(np.ones((subObject_size, subObject_size)), shape=(subObject_size, subObject_size)).multiply(distances)
            # print(distances.toarray())# adjacency = adjacency.multiply(distances)
        else:
            adjacency = adjacency.multiply(distances)
            if False:
                # MST graph.
                adjacency = scipy.sparse.csgraph.minimum_spanning_tree(adjacency)
                adjacency = scipy.sparse.csr_matrix(adjacency).toarray()
                adjacency += adjacency.T - np.diag(adjacency.diagonal())
            else:
                # Delaunay graph.
                adjacency = scipy.sparse.csr_matrix(adjacency).toarray()

        #if is_distance:
        #    # Distance matrix. is it necessary to be normalized?
        #    distances = sklearn.metrics.pairwise.pairwise_distances(points[:,0:2], metric="euclidean", n_jobs=1)
        #    adjacency = adjacency.multiply(distances)

        adjacency = scipy.sparse.csr_matrix(adjacency)
        assert subObject_size  == points.shape[0]
        assert type(adjacency) is scipy.sparse.csr.csr_matrix

        # # 4 collecting the sample: vertice_Geometry,vertice_Fourier,adjacency,label.
        vertices_Geometry.append(Node_features)
        adjacencies.append(adjacency)
        labels.append(label)
        inFIDs.append(k)

    # preprocessing inputs.
    pro_method = True        # to control the m
    if pro_method:
        # standardizing
        if data_type==1:
            # Calculate the mean and std of train dataset, they also will be used to validation and test dataset.
            concatenate_Geometry=np.concatenate(vertices_Geometry,axis=0)
            mean_geometry=concatenate_Geometry.mean(axis=0)
            std_geometry=concatenate_Geometry.std(axis=0)

            if data_type==1:
                file = r'C:\Users\wh\Desktop\gcnn_classification\lib\data\_used_new22.txt'
                file = "./lib/data/_config_22.txt"
                conc = np.vstack((mean_geometry, std_geometry))
                np.savetxt(file, conc, fmt='%.18f')
        if data_type==-1:                                           # for the extra experiment.
            # Import the mean and std of train dataset.
            file = r'C:\Users\wh\Desktop\gcnn_classification\lib\data\_used_new22.txt'
            file = "./lib/data/_config_22.txt"
            conc = np.loadtxt(file)
            mean_geometry, std_geometry = conc[0,:], conc[1,:]
            mean_fourier, std_fourier = conc[0,:], conc[1,:]        # This two parameters are just for fun, do not matter. 
            print("\n========import the mean and std of train dataset from text file========\n")
            #print(mean_geometry)
            #print(std_geometry)

        if True:
            # # The efficiency can be improved by means of vectorization.s
            for i in range(0, len(vertices_Geometry)):
                vertices_shape=np.array((vertices_Geometry[i])).shape
                vertices_Geometry[i] -= np.tile(mean_geometry,vertices_shape[0]).reshape(vertices_shape)
                vertices_Geometry[i] /= np.tile(std_geometry,vertices_shape[0]).reshape(vertices_shape)

                # vertices_shape=np.array((vertices_Fourier[i])).shape
                # vertices_Fourier[i] -= np.tile(mean_fourier,vertices_shape[0]).reshape(vertices_shape)
                # vertices_Fourier[i] /= np.tile(std_fourier,vertices_shape[0]).reshape(vertices_shape)
    else:
        # normalizing, it is not working very well.
        if data_type==1:
            # Calculate the mean and std of train dataset, they also will be used to validation and test dataset.
            concatenate_Geometry=np.concatenate(vertices_Geometry,axis=0)
            mean_geometry=concatenate_Geometry.min(axis=0)
            std_geometry=concatenate_Geometry.max(axis=0)

            file = r'C:\Users\wh\Desktop\gcnn_classification\lib\data\_used2.txt'
            file = "./data/_config_ex.txt"
            conc = np.vstack((mean_geometry, std_geometry))
            np.savetxt(file, conc, fmt='%.18f')

        if not pro_method:
            # # The efficiency can be improved by means of vectorization.s
            for i in range(0, len(vertices_Geometry)):
                vertices_shape=np.array((vertices_Geometry[i])).shape
                vertices_Geometry[i] = (vertices_Geometry[i] - np.tile(mean_geometry,vertices_shape[0]).reshape(vertices_shape))/(std_geometry-mean_geometry)

    # padding.
    # the max number of vertices in a group (sample).
    maxnum_vertices = 128  #max([len(vertices_Geometry[i]) for i in range(0,len(vertices_Geometry))])
    graph_vertices_geo, graph_adjacencies = [], []

    assert len(vertices_Geometry) == len(adjacencies) == len(labels)

    #print(len(vertices_Geometry))
    #print(len(vertices_Geometry[i]))
    #print(np.pad(vertices_Geometry[i], ((0, maxnum_vertices-len(vertices_Geometry[i])),(0,0)), 'constant', constant_values=(0)).shape)
    #exit()

    for i in range(0, len(vertices_Geometry)):
        graph_vertices_geo.append(np.pad(vertices_Geometry[i],
                                              ((0,maxnum_vertices-len(vertices_Geometry[i])),(0,0)),
                                              'constant', constant_values=(0)))
        graph_adjacencies.append(np.pad(adjacencies[i].toarray(),
                                 ((0,maxnum_vertices-adjacencies[i].shape[0]),(0,maxnum_vertices-adjacencies[i].shape[0])),
                                 'constant', constant_values=(0)))
    # collecting.
    graph_vertices_geo     = np.stack(graph_vertices_geo,axis=0).astype(np.float32)       #NSample x NVertices x NFeature
    graph_adjacencies      = np.stack(graph_adjacencies,axis=0).astype(np.float32)        #NSample x NVertices x NVertices
    graph_labels           = np.array(labels).astype(np.int64)                            #NSample x 1
    graph_inFIDs           = np.array(inFIDs).astype(np.int64)                            #NSample x 1
    graph_size             = graph_labels.shape[0]                                        #NSample
    graph_Laplacian        = np.stack([graph.laplacian(scipy.sparse.csr_matrix(A), normalized=True, rescaled=True) for A in graph_adjacencies],axis=0)

    return [graph_vertices_geo,
            graph_Laplacian, 
            graph_labels,
            graph_inFIDs, 
            graph_size,
            mean_geometry, std_geometry]

# load_data('gz1124i.json', [0.6, 0.2, 0.2])