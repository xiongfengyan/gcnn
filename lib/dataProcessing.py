# Copyright 2018 Wuhan Univeristy. All Rights Reserved.
# 2018-12-17

# ==============================================================================
"""Functions for downloading and reading buildings data."""

import json, os, math, datetime

import numpy as np
np.set_printoptions(suppress=True)

import geoutils, geoutils2

import scipy.sparse, scipy.sparse.csgraph, sklearn.metrics, sklearn.decomposition
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def process_data(filename):
    print("Interpreting the {} data".format(filename))
    file = open(filename,'r', encoding='utf-8')
    data = json.load(file)
    feature_size = len(data['features'])
    buildings, labels, inFIDDic = [], [], {}
    for i in range(0, feature_size):
        # Get the attributes.
        label = data['features'][i]['attributes']['type']
        inFID = data['features'][i]['attributes']['inFID']

        # Get density index.
        # This index was computed in ArcMap and stored in Shpfile.
        density = data['features'][i]['attributes']['density']   # nCohesion

        # Get the geometry objects.
        geome_dict  = data['features'][i]['geometry']
        geo_content = geome_dict.get('rings')
        if geo_content == None:
            geo_content = geome_dict.get('paths')
        if geo_content == None:
            print("Please Check the input data.")
            file.close()
            return

        # Just handle the first simple geoobject.
        # if(len(geo_content)>1):
            #print("Multi_Component GeoObject")
            #continue
        building_coords=[] 
        for j in range(0,len(geo_content[0])):
            building_coords.append([geo_content[0][j][0],geo_content[0][j][1]])
        
        if inFIDDic.get(inFID) == None:
            inFIDDic[inFID]=[label, [building_coords], [[density]]]
        else:
            inFIDDic[inFID][1].append(building_coords)
            inFIDDic[inFID][2].append([density])
    file.close()
    interpretingData(inFIDDic)

def interpretingData(inFIDDic):
    #inFiDDic {key,[label,pointlist]}
    if len(inFIDDic) < 1: return None

    process_count, interpretedDic = 0, {}
    for k in inFIDDic:
        # # 0 display the process progress.
        process_count += 1
        if process_count % 200 == 0:
            time_str = datetime.datetime.now().isoformat()
            print("{}: progress {}%...".format(time_str, round(int(process_count*100/len(inFIDDic)),1)))

        # # 2 get the feature vector of vertices.
        #     representing the building object (one vertice) by a feature vector,
        #     Fourer expansion or Geometry description.
        #     the number of buildings (vertices) in one building group (a sample)
        label, subObject_size = inFIDDic[k][0], len(inFIDDic[k][1])
        Node_coords, Node_features = [],[]
        if len(inFIDDic) < 5:
            print("subObject_size: {0}".format(subObject_size))
        if subObject_size < 6 or subObject_size > 128:
            print("debug:      size={0},   ID={1}".format(subObject_size, k))
            continue

        for i in range(0, subObject_size):
            # one building in the sample.
            # if i != 4: continue                   # for debuging
            subObject=inFIDDic[k][1][i]

            [density]=inFIDDic[k][2][i]

            # Calculate the basic indicators of polygon.
            # Geometry descriptors: area, peri, SMBR_area
            [[CX,CY],area,peri] = geoutils.get_basic_parametries_of_Poly(subObject)

            compactness     = area / peri                               # area / math.pow(0.282*peri, 2)
            OBB, SMBR_area  = geoutils.mininumAreaRectangle(subObject)  # 
            orientation     = OBB.Orientation()
            length_width    = OBB.e1 / OBB.e0 if OBB.e0 > OBB.e1 else OBB.e0 / OBB.e1
            area_radio      = area / (OBB.e1 * OBB.e0 * 4)
            # print("area={}, peri={}, SMBR_area={}".format(area, peri, SMBR_area))

            # Three basic indices. Faster
            # geo_features = [orientation, area, length_width, area_radio, compactness]
            geo_features = [orientation, area, length_width, compactness]

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

                # debug information
                # import matplotlib.pyplot as plt
                # plt.plot(uniform_coords[:, 0], uniform_coords[:, 1], 'o')
                # plt.plot(uniform_coords[:, 0], uniform_coords[:, 1], 'b--', lw=1)
                # plt.plot(uniform_coords[convexHull.vertices, 0], uniform_coords[convexHull.vertices, 1], 'o')
                # plt.plot(uniform_coords[convexHull.vertices, 0], uniform_coords[convexHull.vertices, 1], 'r--', lw=2)
                # enddebug

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



                # --------------------------------------------
                # PART TWO: ORIENTATION INDICES
                # LONGEDGE_orien, SMBR_orien, WEIGHT_orien
                # --------------------------------------------

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
                

                '''# debug information.
                print(longedge_a * from_longedge[0] + longedge_b * from_longedge[1] + longedge_c)
                print(longedge_a * to_longedge[0] + longedge_b * to_longedge[1] + longedge_c)
                print("from_longedge={}, to_longedge={}".format(from_longedge, to_longedge))
                print("longedge_a={}, longedge_b={},  longedge_c={}".format(longedge_a, longedge_b, longedge_c))
                print("longedge_c={}, up_offset={},  down_offset={}".format(longedge_c, up_offset, down_offset))

                plt.plot([from_longedge[0], to_longedge[0]], [from_longedge[1], to_longedge[1]], 'r-', lw=2)
                plt.plot([from_longedge[0], to_longedge[0]], [-(from_longedge[0]*longedge_a+up_offset)/longedge_b, -(to_longedge[0]*longedge_a+up_offset)/longedge_b], 'b-', lw=2)
                plt.plot([from_longedge[0], to_longedge[0]], [-(from_longedge[0]*longedge_a+down_offset)/longedge_b, -(to_longedge[0]*longedge_a+down_offset)/longedge_b], 'b-', lw=2)
                # plt.plot([from_longedge[0], to_longedge[0]], [from_longedge[0]+down_offset-longedge_c, to_longedge[1]+down_offset-longedge_c], 'g-', lw=2)
                # enddebug'''

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

                '''# debug information.
                plt.plot([-math.sin(WALL_orien)*peri*.5/math.pi, math.sin(WALL_orien)*peri*.5/math.pi],\
                         [-math.cos(WALL_orien)*peri*.5/math.pi, math.cos(WALL_orien)*peri*.5/math.pi], 'r--', lw=2)
                plt.plot([-math.sin(WALL_orien+math.pi/2)*peri*.5/math.pi, math.sin(WALL_orien+math.pi/2)*peri*.5/math.pi],\
                         [-math.cos(WALL_orien+math.pi/2)*peri*.5/math.pi, math.cos(WALL_orien+math.pi/2)*peri*.5/math.pi], 'r--', lw=2)

                plt.plot([-math.sin(WEIGHT_orien)*peri*.5/math.pi, math.sin(WEIGHT_orien)*peri*.5/math.pi],\
                         [-math.cos(WEIGHT_orien)*peri*.5/math.pi, math.cos(WEIGHT_orien)*peri*.5/math.pi], 'b-', lw=2)
                plt.plot([-math.sin(WEIGHT_orien+math.pi/2)*peri*.5/math.pi, math.sin(WEIGHT_orien+math.pi/2)*peri*.5/math.pi],\
                         [-math.cos(WEIGHT_orien+math.pi/2)*peri*.5/math.pi, math.cos(WEIGHT_orien+math.pi/2)*peri*.5/math.pi], 'b-', lw=2)
                #enddebug'''

                # print("LONGEDGE_orien={}, SMBR_orien={}, WALL_orien={}, WEIGHT_orien={}".format(LONGEDGE_orien, SMBR_orien, WALL_orien, WEIGHT_orien))


                # Part three. Shape indicators: concavity, elongation, eccentricity, circularity, ellipticity, shape_index, fractal_dim, edge_count
                #                      IPQ_compa, RIT_compa, RIC_compa, GIB_compa, DCM_compa, BOT_compa, BOY_compa
                # IPQ_compa, RIT_compa, RIC_compa, GIB_compa = 4*math.pi*area/(peri*peri), area/peri, 2*math.sqrt(math.pi*area)/peri, 4*area/LONG_chord



                # --------------------------------------------
                # PART THREE: SHAPE INDICES
                # Diameter-Perimeter-Area- measurements
                # --------------------------------------------

                RIC_compa, IPQ_compa, FRA_compa = area/peri, 4*math.pi*area/(peri*peri), 1-math.log(area)*.5/math.log(peri)
                GIB_compa, Div_compa = 2*math.sqrt(math.pi*area)/LONG_chord, 4*area/(LONG_chord*peri)



                # --------------------------------------------
                # PART FOUR: SHAPE INCICES
                # Related shape
                # --------------------------------------------

                # fit_Ellipse = geoutils2.fitEllipse(np.array(uniform_coords)[:,0], np.array(uniform_coords)[:,1])
                # ellipse_axi = geoutils2.ellipse_axis_length(fit_Ellipse)
                # elongation, ellipticity, concavity = length_width, ellipse_axi[0]/ellipse_axi[1] if ellipse_axi[1] != 0 else 1, area/convexHull.area
                elongation, ellipticity, concavity = length_width, LONG_width/LONG_chord, area/convexHull.area


                radius, standard_circle, enclosing_circle = math.sqrt(area / math.pi), [], geoutils2.make_circle(uniform_coords)
                for j in range(0, 60):
                    standard_circle.append([math.cos(2*math.pi*j/60)*radius, math.sin(2*math.pi*j/60)*radius])

                standard_intersection = Polygon(uniform_coords).intersection(Polygon(standard_circle))
                standard_union = Polygon(uniform_coords).union(Polygon(standard_circle))


                '''# debug information
                print("enclosing_circle={}".format(enclosing_circle))
                if enclosing_circle is not None:
                    xs, ys = [], []
                    for j in range(0, 100):
                        xs.append(enclosing_circle[0] + math.cos(2*math.pi*j/100)*enclosing_circle[2])
                        ys.append(enclosing_circle[1] + math.sin(2*math.pi*j/100)*enclosing_circle[2])
                    xs.append(xs[0])
                    ys.append(ys[0])
                    plt.plot(xs, ys, 'k--', lw=1)
                # enddebug'''

                '''# debug information
                # Note: if the geometry is multipolygon, it will go wrong.
                print("standard_intersection_area={}".format(standard_intersection.area))
                print("standard_union_area={}".format(standard_union.area))
                
                plt.plot(np.array(standard_intersection.exterior.coords)[:, 0], np.array(standard_intersection.exterior.coords)[:, 1], 'o')
                plt.plot(np.array(standard_intersection.exterior.coords)[:, 0], np.array(standard_intersection.exterior.coords)[:, 1], 'b--', lw=1)

                plt.plot(np.array(standard_union.exterior.coords)[:, 0], np.array(standard_union.exterior.coords)[:, 1], 'o')
                plt.plot(np.array(standard_union.exterior.coords)[:, 0], np.array(standard_union.exterior.coords)[:, 1], 'y--', lw=1)
                # enddebug'''

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

                # concavity, elongation, eccentricity, circularity = area/convexHull.area, length_width, 4*area/(LONG_chord*peri), peri*peri/area
                #ellipticity, shape_index, fractal_dim, edge_count = LONG_width/LONG_chord, peri/math.sqrt(4*math.pi*area), 2*math.log(peri)/math.log(area), uniform_size-1

                #plt.gca().set_aspect(1)
                #plt.show()



                # --------------------------------------------
                # PART FIVE: SHAPE INCICES
                # Interior grid point properties
                # This three indices are calculated by the Shape Metrices Tool written by Jason Parent.
                # --------------------------------------------


                # --------------------------------------------
                # PART SIX: SHAPE INCICES
                # Dispersion of elements / components of area
                # --------------------------------------------
                M02, M20, M11 = 0, 0, 0
                for j in range(0, uniform_size-1):
                    M02 += (uniform_coords[j][1])*(uniform_coords[j][1])
                    M20 += (uniform_coords[j][0])*(uniform_coords[j][0])
                    M11 += (uniform_coords[j][0])*(uniform_coords[j][1])

                Eccentricity = ((M02+M20)*(M02+M20)+4*M11)/area

                # geometry transform to make the area consistant.
                ### XA,XB = [j/math.sqrt(area_) for j in XA],[j/math.sqrt(area_) for j in XB]
                ### YA,YB = [j/math.sqrt(area_) for j in YA],[j/math.sqrt(area_) for j in YB]

                # # 2.1 compositing the descriptors as Geometry feature vecters.
                geo_features = [area, peri, LONG_chord, MEAN_radius, \
                                SMBR_orien, LONGEDGE_orien, BISSECTOR_orien, WEIGHT_orien, \
                                RIC_compa, IPQ_compa, FRA_compa, GIB_compa, Div_compa, \
                                elongation, ellipticity, concavity, DCM_index, BOT_index, BOY_index, \
                                M11, Eccentricity, \
                                density]

            Node_coords.append([CX, CY])
            Node_features.append(geo_features)
        interpretedDic[k] = [label, Node_coords, Node_features]
    
    with open('./data/ac1201i.json','w') as json_file:
        json.dump(interpretedDic, json_file, indent = 2, ensure_ascii = False)

process_data('./data/ac1201.json')