# Copyright 2018@Wuhan Univeristy. All Rights Reserved.
#
# =====================================================
# Figure
import json,os,math
import os, time, datetime

import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
import sklearn.metrics, scipy.sparse, scipy.sparse.csgraph
from tensorflow.contrib import learn

from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon

import models, graph, coarsening, geoutils, geoutils2
import matplotlib.pyplot as plt

def load_data(filename):
    print("Parsing the {} data".format(filename))
    file=open(filename,'r',encoding='utf-8')
    data=json.load(file)
    feature_size=len(data['features'])
    buildings, labels=[],[]
    inFIDDic={}
    for i in range(0, feature_size):
        # Get the attributes.
        label=data['features'][i]['attributes']['type']
        inFID=data['features'][i]['attributes']['inFID']

        # Get indices.
        density = data['features'][i]['attributes']['density']   # nCohesion

        # Get the geometry objects.
        geome_dict=data['features'][i]['geometry']
        geo_content=geome_dict.get('rings')
        if geo_content==None:
            geo_content=geome_dict.get('paths')
        if geo_content==None:
            print("Please Check the input data.")
            file.close()
            return [],[]
        # Just handle the first simple geoobject.
        # if(len(geo_content)>1):
            #print("Multi_Component GeoObject")
            #continue
        building_coords=[]
        for j in range(0,len(geo_content[0])):
            building_coords.append([geo_content[0][j][0],geo_content[0][j][1]])
        
        if inFIDDic.get(inFID) == None:
            inFIDDic[inFID]=[label,[building_coords],[[density]]]
        else:
            inFIDDic[inFID][1].append(building_coords)
            inFIDDic[inFID][2].append([density])
    file.close()
    return fGraph(inFIDDic)

def fGraph(inFIDDic, series_size=3, is_distance=True):
    # # 1 get the label of this sample.
    k = list(inFIDDic.keys())[0]
    #print(inFIDDic)
    inFIDDic[k][0]
    label=1 if inFIDDic[k][0]==3 else 0
    # # 2 get the feature vector of vertices.
    #     representing the building object (one vertice) by a feature vector,
    #     Fourer expansion or Geometry description.
    #     the number of buildings (vertices) in one building group (a sample)
    subObject_size=len(inFIDDic[k][1])
    XY_coords, vertice, coords = [], [], []


    for i in range(0, subObject_size):

        # one building in the sample.
        subObject=inFIDDic[k][1][i]

        [density]=inFIDDic[k][2][i]

        # Calculate the basic indicators of polygon.
        # Geometry descriptors: area, peri, SMBR_area
        [[CX,CY],area,peri] = geoutils.get_basic_parametries_of_Poly(subObject)
        XY_coords.append([CX, CY, i])

        compactness  = area / math.pow(0.282*peri, 2)
        OBB, SMBR_area = geoutils.mininumAreaRectangle(subObject)
        orientation  = OBB.Orientation()
        length_width = OBB.e1 / OBB.e0 if OBB.e0 > OBB.e1 else OBB.e0 / OBB.e1
        area_radio   = area / (OBB.e1 * OBB.e0 * 4)
        # print("area={}, peri={}, SMBR_area={}".format(area, peri, SMBR_area))

        # Five basic indices. Faster
        geometry_vector=[orientation, area, length_width, area_radio, compactness]

        # More indices and more slowly.
        if True:
            # preparatory work
            uniform_coords = np.array([[(j[0]-CX), (j[1]-CY)] for j in subObject])
            uniform_size = len(uniform_coords)
            # Closing the polygon.
            if uniform_coords[0][0] - uniform_coords[uniform_size-1][0] != 0 or uniform_coords[0][1] - uniform_coords[uniform_size-1][1] != 0:
                print('Closing!')
                uniform_coords.append(uniform_coords[0])

            # Part One. Size indicators: CONVEX_area, MEAN_radius, LONG_chord
            convexHull = ConvexHull(uniform_coords)
            CONVEX_area = convexHull.area

            sum_radius, size_radius, MEAN_radius, LONG_chord = 0, 0, 0, 0
            for j in range(0, uniform_size-1):
                sum_radius += math.sqrt(uniform_coords[j][0]*uniform_coords[j][0]+uniform_coords[j][1]*uniform_coords[j][1])
                size_radius += 1
            if size_radius != 0:
                MEAN_radius = sum_radius / size_radius

            pairwise_distances, index_j, index_h = sklearn.metrics.pairwise.pairwise_distances(uniform_coords[convexHull.vertices], metric="euclidean", n_jobs=1), 0, 0
            for j in range(0, len(pairwise_distances)):
                for h in range(j, len(pairwise_distances)):
                    if (pairwise_distances[j, h] > LONG_chord):
                        LONG_chord, index_j, index_h = pairwise_distances[j, h], j, h

            SECOND_chord, index_p, index_q = 0, 0, 0
            for j in range(0, len(pairwise_distances)):
                for h in range(j, len(pairwise_distances)):
                    if pairwise_distances[j, h] > SECOND_chord:
                        if j != index_j and h != index_h:
                            SECOND_chord, index_p, index_q = pairwise_distances[j, h], j, h




            # Part two. Orientation indicators: LONGEDGE_orien, SMBR_orien, WEIGHT_orien

            from_longedge, to_longedge = uniform_coords[convexHull.vertices[index_j]], uniform_coords[convexHull.vertices[index_h]]
            LONGEDGE_orien = abs(math.atan2(from_longedge[0]-to_longedge[0], from_longedge[1]-to_longedge[1]))
            
            from_secondedge, to_secondedge = uniform_coords[convexHull.vertices[index_p]], uniform_coords[convexHull.vertices[index_q]]
            SENCONDEDGE_orien = abs(math.atan2(from_secondedge[0]-to_secondedge[0], from_secondedge[1]-to_secondedge[1]))


            #LONGEDGE_dis = math.sqrt((from_longedge[0]-to_longedge[0])*(from_longedge[0]-to_longedge[0]) + (from_longedge[1]-to_longedge[1])*(from_longedge[1]-to_longedge[1]))
            SECONDEDGE_dis = math.sqrt((from_secondedge[0]-to_secondedge[0])*(from_secondedge[0]-to_secondedge[0]) + (from_secondedge[1]-to_secondedge[1])*(from_secondedge[1]-to_secondedge[1]))
            
            BISSECTOR_orien = (LONGEDGE_orien*LONG_chord+SENCONDEDGE_orien*SECONDEDGE_dis) / (LONG_chord+SECONDEDGE_dis)
            # print("LONG_width={}, LONGEDGE_dis={}".format(LONG_width, LONGEDGE_dis))



            SMBR_orien, WALL_orien, WEIGHT_orien = orientation, 0, 0
            
            # Calculate vertical width agaist long cord.
            # line equation:
            longedge_a, longedge_b, longedge_c = geoutils2.get_equation(from_longedge, to_longedge)
            LONG_width, up_offset, down_offset = 0, longedge_c, longedge_c
            for j in range(0, uniform_size-1):
                crossing_product = longedge_a*uniform_coords[j][0]+longedge_b*uniform_coords[j][1]
                if crossing_product + up_offset < 0:
                    up_offset = -crossing_product
                if crossing_product + down_offset > 0:
                    down_offset = -crossing_product
            longedge_square = math.sqrt(longedge_a*longedge_a+longedge_b*longedge_b)
            if longedge_square == 0:
                LONG_width = abs(up_offset-down_offset)
            else:
                LONG_width = abs(up_offset-down_offset)/longedge_square

            edge_orien_weight, edge_length_sun, edge_tuple, candidate_max = 0, 0, [], 0
            for j in range(0, uniform_size-1):
                dx, dy = uniform_coords[j+1][0]-uniform_coords[j][0], uniform_coords[j+1][1]-uniform_coords[j][1]
                edge_orien = (math.atan2(dx, dy) + math.pi) % (math.pi/2.0)
                # edge_orien = (math.atan2(dx, dy) + 2*math.pi) % math.pi
                # edge_orien = math.atan2(dx, dy)
                edge_length = math.sqrt(dx*dx + dy*dy)

                edge_orien_weight += edge_length*edge_orien
                edge_length_sun += edge_length

                edge_tuple.append([edge_orien, edge_length])
                # add test code.
                # print("edge_length={},  edge_orien={}".format(edge_length, edge_orien*180/math.pi))
            WALL_orien = edge_orien_weight / edge_length_sun

            for j in range(0, 90):
                candidate_orien, candidate_weight = j*math.pi/180, 0
                for j in range(0, len(edge_tuple)):
                    if abs(edge_tuple[j][0]-candidate_orien) < math.pi/24:
                        candidate_weight += (math.pi/24 - abs(edge_tuple[j][0]-candidate_orien))*edge_tuple[j][1]/(math.pi/24)
                if candidate_weight > candidate_max:
                    candidate_max, WEIGHT_orien = candidate_weight, candidate_orien



            # Part three. shape indices: Diameter-Perimeter-Area- measurements

            RIC_compa, IPQ_compa, FRA_compa = area/peri, 4*math.pi*area/(peri*peri), 1-math.log(area)*.5/math.log(peri)
            GIB_compa, Div_compa = 2*math.sqrt(math.pi*area)/LONG_chord, 4*area/(LONG_chord*peri)



            # Part four. shape indices: Related shape

            # fit_Ellipse = geoutils2.fitEllipse(np.array(uniform_coords)[:,0], np.array(uniform_coords)[:,1])
            # ellipse_axi = geoutils2.ellipse_axis_length(fit_Ellipse)
            # elongation, ellipticity, concavity = length_width, ellipse_axi[0]/ellipse_axi[1] if ellipse_axi[1] != 0 else 1, area/convexHull.area
            elongation, ellipticity, concavity = length_width, LONG_width/LONG_chord, area/convexHull.area


            radius, standard_circle, enclosing_circle = math.sqrt(area / math.pi), [], geoutils2.make_circle(uniform_coords)
            for j in range(0, 60):
                standard_circle.append([math.cos(2*math.pi*j/100)*radius, math.sin(2*math.pi*j/100)*radius])

            standard_intersection = Polygon(uniform_coords).intersection(Polygon(standard_circle))
            standard_union = Polygon(uniform_coords).union(Polygon(standard_circle))

            DCM_index = area / (math.pi*enclosing_circle[2]*enclosing_circle[2])
            BOT_index = 1-standard_intersection.area/area

            closest_length, closest_sun, closest_size, BOY_measure = [], 0, 0, 0
            for j in range(0, 40):
                x, y = math.cos(2*math.pi*j/40)*peri, math.sin(2*math.pi*j/40)*peri
                closest_point, is_test = geoutils2.find_intersection(uniform_coords, [x, y])
                if is_test:
                    print("k={},  i={},  j={}".format(k, i, j))
                    # plt.plot([0, closest_point[0]], [0, closest_point[1]]) # debug

                if closest_point is not None:
                    # plt.plot([0, closest_point[0]], [0, closest_point[1]]) # debug
                    closest_length.append(math.sqrt(closest_point[0]*closest_point[0]+closest_point[1]*closest_point[1]))
                    closest_sun += math.sqrt(closest_point[0]*closest_point[0]+closest_point[1]*closest_point[1])
                    closest_size += 1
                #else:
                #    print("Maybe the centerpoint is not in the polygon.")
            for j in closest_length:
                BOY_measure += abs(100*j/closest_sun-100/closest_size)
            BOY_index = 1-BOY_measure/200
            #print("BOY_index={}".format(BOY_index))

            # Part six. shape indices: Dispersion of elements / components of area
            M02, M20, M11 = 0, 0, 0
            for j in range(0, uniform_size-1):
                M02 += (uniform_coords[j][1])*(uniform_coords[j][1])
                M20 += (uniform_coords[j][0])*(uniform_coords[j][0])
                M11 += (uniform_coords[j][0])*(uniform_coords[j][1])

            Eccentricity = ((M02+M20)*(M02+M20)+4*M11)/area


            geometry_vector = [CX, CY, area, peri, LONG_chord, MEAN_radius, \
                               SMBR_orien, LONGEDGE_orien, BISSECTOR_orien, WEIGHT_orien, \
                               RIC_compa, IPQ_compa, FRA_compa, GIB_compa, Div_compa, \
                               elongation, ellipticity, concavity, DCM_index, BOT_index, BOY_index, \
                               M11, Eccentricity, \
                               density]
            geometry_vector = [CX, CY, area, peri, MEAN_radius, \
                               SMBR_orien, WEIGHT_orien, \
                               IPQ_compa, FRA_compa, elongation, concavity, BOT_index, \
                               density]
        vertice.append(geometry_vector)
        coords.append(subObject)

    # # 3 get the adjacency graph of the building group (one sample).
    # KNN, MST, Delaunay = 1, 2, 3.
    vertice = np.array(vertice)
    points = np.array(XY_coords)
    adjacency = np.zeros((subObject_size, subObject_size))
    # print(points[:,0:2])
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
    
    distances = sklearn.metrics.pairwise.pairwise_distances(points[:,0:2], metric='euclidean')
    if False:
        # Distance matrix. is it necessary to be normalized?    
        idx = np.argsort(distances)[:, 1:1+1]
        distances.sort()
        distances = graph.adjacency(distances[:, 1:1+1], idx)
        adjacency = scipy.sparse.coo_matrix(np.ones((subObject_size, subObject_size)), shape=(subObject_size, subObject_size)).multiply(distances)
        print(distances.toarray())# adjacency = adjacency.multiply(distances)
    else:
        adjacency = adjacency.multiply(distances)
        if False:
            adjacency = scipy.sparse.csgraph.minimum_spanning_tree(adjacency)
            adjacency = scipy.sparse.csr_matrix(adjacency).toarray()
            adjacency += adjacency.T - np.diag(adjacency.diagonal())
            # print(adjacency)
        else:
            # distances = sklearn.metrics.pairwise.pairwise_distances(points[:,0:2], metric="euclidean", n_jobs=1)
            adjacency = scipy.sparse.csr_matrix(adjacency).toarray()

    if False:
        file = r'C:\Users\Administrator\Desktop\ahtor\thesis\gcnn\data\_used_bk.txt'
        file = "./data/_config_22.txt"
        conc = np.loadtxt(file)
        #print((vertice[0,3] - conc[0,1]) / conc[1,1])
        vertice_shape = vertice[:,2:].shape
        vertice[:,2:] -= np.tile(conc[0,:], vertice_shape[0]).reshape(vertice_shape)
        vertice[:,2:] /= np.tile(conc[1,:], vertice_shape[0]).reshape(vertice_shape)

    if len(vertice) < 128 and False:
        vertice   = np.pad(vertice, ((0, 128-len(vertice)),(0,0)), 'constant', constant_values=(0))
        adjacency = np.pad(adjacency, ((0, 128-adjacency.shape[0]),(0, 128-adjacency.shape[0])), 'constant', constant_values=(0))

    laplacian = graph.laplacian(scipy.sparse.csr_matrix(adjacency), normalized=True, rescaled=True)
    print(vertice.shape)
    print(adjacency.shape)
    print(laplacian.shape)
    return coords, np.array(vertice), np.array(adjacency), laplacian, np.array(label)

def monomial(self, vertices, adjacencies, Fout, K):
    N, M, Fin = vertices.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    assert N == adjacencies.shape[0]
    assert M == adjacencies.shape[1] == adjacencies.shape[2]
    W = self._weight_variable([Fin*K, Fout], regularization=True)    # Fin*K X Fout # False
    X=[]
    # The efficiency can be improved by means of vectorization.
    for i in range(0, N):
        Li = adjacencies[i]                                   # M x M
        # Transform to monomial basis
        Xi_0 = vertices[i]                                    # M x Fin
        Xi   = tf.expand_dims(Xi_0, 0)                        # 1 x M x Fin
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)                        # 1 x M x Fin
            return tf.concat([x, x_], axis=0)                 # K x M x Fin    
        for k in range(1, K):
            Xi_0 = tf.matmul(Li, Xi_0)    # M x Fin*N
            Xi   = concat(Xi, Xi_0)
        Xi = tf.reshape(Xi, [K, M, Fin])                      # K x M x Fin
        Xi = tf.transpose(Xi, [1, 2, 0])                      # M x Fin x K
        Xi = tf.reshape(Xi, [M, Fin*K])                       # M x Fin*K
        Xi = tf.matmul(Xi, W)                                 # [M x Fin*K] x[Fin*k x Fout] = [M X Fout]
        X.append(Xi)
    return tf.reshape(X, [N, M, Fout])
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

def shownodedomains2D(ax, C, V, A, f, min_F=0, max_F=1):
    #print(vertice[:,feature_])
    # The render 
    size = len(C)
    verts = []
    for i in range(0, size):
        geo = np.array(C[i])
        verts.append(list(zip(geo[:,0], geo[:,1])))
        for j in range(0, size):
            if A[i][j]>0:
                ax.plot([V[i][0], V[j][0]], [V[i][1], V[j][1]], color='r', linewidth=0.01, zorder=1, alpha=0.3)
    
    cm2 = plt.get_cmap("BuGn")    #BuGn
    #cols2 = [cm2(float(i)/(size+int(size/1.25))) for i in range(size+int(size/1.25))][-size:]
    #print("1: f={}".format(f))
    #d = f[:]
    #idx = np.argsort(d)[:]
    min_F, max_F = np.min(f), np.max(f)
    if min_F == max_F: min_F, max_F = 0, 1
    if True:
        for i in range(0, size):
            # i denotes the sorted index, and idx[i] denotes the orignal index.
            a, b, r = V[i,0], V[i,1], 3.0
            theta = np.arange(0, 2*np.pi, 0.5)
            x = a + r * np.cos(theta)
            y = b + r * np.sin(theta)
            #ax.plot(x, y, color='w', linewidth=2, zorder=2)
            ax.fill(x, y, c=cm2(0.4*(f[i]-min_F)/(max_F-min_F)+0.6), zorder=2)
            # ax.text(V[idx[i],0], V[idx[i],1], i, zorder=15)
    else:
        for i in range(0, size):
            # i denotes the sorted index, and idx[i] denotes the orignal index.
            a, b, r = V[i,0], V[i,1], 9.0
            ax.scatter(a, b, s=3.0, c=cm2(0.4*(f[i]-min_F)/(max_F-min_F)+0.6), zorder=2)
            # ax.text(V[idx[i],0], V[idx[i],1], i, zorder=15)
    #print("2: f={}".format(f))

    ax.set_axis_off()
    ax.set_aspect(1)
    #poly = PolyCollection(verts, facecolors=[cc('0.2')]*size, alpha=0.1)
    #ax.add_collection3d(poly, zs=[0]*size, zdir='z')
    #ax.scatter(V[:,0], V[:,1],  s = 1, c=[cc('r')]*size)
    # ax.bar(V[:,0], f, zs=V[:,1], zdir='y', color=[cc('b') if f[i]>0 else cc('#000000') for i in range(0,size)])

    # ax.set_xlim3d(min([min(np.array(i)[:,0]) for i in C]),max([max(np.array(i)[:,0]) for i in C]))
    # ax.set_ylim3d(min([min(np.array(i)[:,1]) for i in C]),max([max(np.array(i)[:,1]) for i in C]))

def showfourierdomains(bx, lamb, fourier):
    #bx.set_ylim([-0.075,1.250])  
    #print("len:{0}  lamb={1}".format(len(lamb), lamb))
    #print("len:{0}  fourier={1}".format(len(fourier), fourier))
    bx.plot(lamb, fourier, linewidth = 10, zorder=2)
    #bx.scatter(lamb, fourier, marker = 'o', color = 'w', s = 120, zorder=4)
    #bx.scatter(lamb, fourier, marker = 'o', color = 'r', s = 200, zorder=3)

    #for i in range(len(lamb)):
    #    bx.plot([lamb[i],lamb[i]], [fourier[i], -3], 'r--', lw=0.5)

    #bx.set_xlim(-1.12, 0.75)
    #bx.set_ylim(-2.1, 3.2)

def show2D(C, V, A, L, F, fignumber, filename='', nodedomain=True, fourierdomain=True):
    feature_num = int(F.shape[1])
    # print("feature_num={}".format(feature_num))
    # print("F={}".format(F))
    lamb, U = np.linalg.eigh(L)
    T=U.T

    min_F, max_F = np.min(F), np.max(F)
    #print("min_F={}, max_F={}".format(min_F, max_F))
    if max_F==min_F:
        min_F, max_F=0, 1
    fig = plt.figure(fignumber, figsize=(10,48))
    for i in range(feature_num):
        ax = fig.add_subplot(feature_num, 2, 2*i+1)
        bx = fig.add_subplot(feature_num, 2, 2*i+2)
        shownodedomains2D(ax, C, V, A, F[:,i], min_F, max_F)

        fourier = T.dot(F[:,i])
        #print(fourier)
        #print(lamb)
        showfourierdomains(bx, lamb, fourier)
    fig.tight_layout()
    fig.savefig('../graphics/'+filename+str(fignumber)+'.pdf')
    #plt.show()

filename='ac1202'
coords, vertice, adjacency, laplacian, label = load_data('./data/'+filename+'.json')
assert len(adjacency) == len(vertice)
size = len(adjacency)
#print(len(adjacency))
import random


# Code 2: Figure 13. Activation maps    
# fourier graph. tested by grid001
show2D(coords, vertice[:,0:2], adjacency, laplacian, vertice[:,2:], 1, filename)
# plt.show()

import tensorflow as tf  
import numpy as np

varies = {}
with tf.Session() as sess:
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(path)
    param_path = os.path.join(path, 'checkpoints', 'fullmap11=1')
    print(param_path)
    check_path = tf.train.latest_checkpoint(param_path)
    model_path = check_path+'.meta'

    new_saver  = tf.train.import_meta_graph(model_path)               #load graph
    new_saver.restore(sess, tf.train.latest_checkpoint(param_path))   #find the newest training result  
    for v in tf.trainable_variables():
        v_4d = np.array(sess.run(v)).astype('float64')                #get the real parameters
        varies[v.name[:-2]]=v_4d
        print(v.name[:-2], ': ', v_4d.shape)

    conv1_tensor = monomial(vertice[:,2:], laplacian, varies['conv1/weights'], 16, 3)
    conv1_vertice__ = np.array(sess.run(conv1_tensor)).astype('float64')
    conv1_tensor = tf.nn.relu(conv1_tensor + varies['conv1/bias'].reshape(varies['conv1/bias'].shape[1], varies['conv1/bias'].shape[2]))
    conv1_vertice = np.array(sess.run(conv1_tensor)).astype('float64')

    conv2_tensor = monomial(conv1_vertice, laplacian, varies['conv2/weights'], 16, 3)
    conv2_vertice__ = np.array(sess.run(conv2_tensor)).astype('float64')
    conv2_tensor = tf.nn.relu(conv2_tensor + varies['conv2/bias'].reshape(varies['conv2/bias'].shape[1], varies['conv2/bias'].shape[2]))
    conv2_vertice = np.array(sess.run(conv2_tensor)).astype('float64')

    conv3_tensor = monomial(conv2_vertice, laplacian, varies['conv3/weights'], 16, 3)
    conv3_vertice__ = np.array(sess.run(conv3_tensor)).astype('float64')
    conv3_tensor = tf.nn.relu(conv3_tensor + varies['conv3/bias'].reshape(varies['conv3/bias'].shape[1], varies['conv3/bias'].shape[2]))
    conv3_vertice = np.array(sess.run(conv3_tensor)).astype('float64')

    conv4_tensor = monomial(conv3_vertice, laplacian, varies['conv4/weights'], 16, 3)
    conv4_vertice__ = np.array(sess.run(conv4_tensor)).astype('float64')
    conv4_tensor = tf.nn.relu(conv4_tensor + varies['conv4/bias'].reshape(varies['conv4/bias'].shape[1], varies['conv4/bias'].shape[2]))
    conv4_vertice = np.array(sess.run(conv4_tensor)).astype('float64')

    conv5_tensor = monomial(conv4_vertice, laplacian, varies['conv5/weights'], 16, 3)
    conv5_vertice__ = np.array(sess.run(conv5_tensor)).astype('float64')
    conv5_tensor = tf.nn.relu(conv5_tensor + varies['conv5/bias'].reshape(varies['conv5/bias'].shape[1], varies['conv5/bias'].shape[2]))
    conv5_vertice = np.array(sess.run(conv5_tensor)).astype('float64')

    conv5_tensor = tf.reshape(conv5_tensor, [1, conv5_tensor.shape[0]*conv5_tensor.shape[1]])
    fc1_tensor = tf.nn.relu(tf.matmul(conv5_tensor, varies['fc1/weights']) + varies['fc1/bias'])

#
#    fc1_vertice = np.array(sess.run(fc1_tensor)).astype('float64')
#    print(fc1_vertice)

#    def show_FC(fc1_vertice):
#        #fc1_vertice = fc1_vertice[0].toarray()
#        fc1_vertice = fc1_vertice.min(axis=0)
#        fig = plt.figure()
#        ax = fig.add_subplot(1,1,1)

#        size = len(fc1_vertice)
#        cm1 = plt.get_cmap("RdBu")
#        cols1 = [cm1(float(i)/(size+int(size/1.25))) for i in range(size+int(size/1.25))][-size:]

#        for i in range(0, size):
#            a, b, r = 0, 5*i, 2.0
#            theta = np.arange(0, 2*np.pi, 0.01)
#            x = a + r * np.cos(theta)
#            y = b + r * np.sin(theta)
#            ax.plot(x, y, color='w', linewidth=2, zorder=2+i)

#            index = int((fc1_vertice[i]-fc1_vertice.min(axis=0))/(fc1_vertice.max(axis=0)-fc1_vertice.min(axis=0))*size)
#            if (index == size):
#                index = index-1
#            if (index < 0):
#                index = 0
#            ax.fill(x, y, c=cols1[index], zorder=2+i)
#            #ax.text(V[i,0], V[i,1], i, zorder=15)
#        #ax.set_axis_off()
#        ax.set_aspect(1)
#        plt.show()
#    show_FC(fc1_vertice)
#

    logits_tensor = tf.nn.relu(tf.matmul(fc1_tensor, varies['logits/weights']) + varies['logits/bias'])
    
    probabilities = tf.nn.softmax(logits_tensor)
    prediction = tf.argmax(logits_tensor, axis=1)
    
    print(np.array(sess.run(probabilities)).astype('float64'))
    print(np.array(sess.run(prediction)).astype('int32'))

    if True:
        show2D(coords, vertice[:,0:2], adjacency, laplacian, conv1_vertice, 2, filename)
        show2D(coords, vertice[:,0:2], adjacency, laplacian, conv2_vertice, 3, filename)
        show2D(coords, vertice[:,0:2], adjacency, laplacian, conv3_vertice, 4, filename)
        show2D(coords, vertice[:,0:2], adjacency, laplacian, conv4_vertice, 5, filename)
        show2D(coords, vertice[:,0:2], adjacency, laplacian, conv5_vertice, 6, filename)

    #show3D(coords, vertice[:,0:2], adjacency, laplacian, conv1_vertice, 2)

#plt.show()
#plt.savefig(format='png', dpi=300)
