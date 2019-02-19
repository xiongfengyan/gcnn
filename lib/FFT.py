# Copyright 2018 @ Wuhan Univeristy. All Rights Reserved.
# This is an implementation of Fourier descriptor for geometric shape.
# 2018-03-22
#
# =====================================================
import math, json
from scipy import integrate

import numpy as np
import matplotlib.pyplot as plt

import geoutils

# Get the value from a piecewise linear function for a given value.
# Args:
#   X=[0,1,2,3,4]: X coordinates in ascending order. 
#   Y=[0,1,1,2,0]: Y coordinates.
#               x: A given value.
# Outputs:
#   return the functional value.
def piecewise_function(X,Y,x):
	if x < X[0] or x > X[len(X)-1]:
		print('ILLEGAL_ARGUMENT')
		return 0;
	i=1
	while i <= len(X)-1:
		if(x<=X[i]):
			y=(Y[i]-Y[i-1])*(x-X[i])/(X[i]-X[i-1])+Y[i]
			return y
		else:
			i=i+1
	return 0

# Adjust the start point and direction of object A.
# Args:
#   A=[(x1,y1),(x2,y2),...,(xn,yn)]: The input geometry object.
# Output:
#   A=[(x1,y1),(x2,y2),...,(xn,yn)]: The output adjusted geometry object.
def adjust_StartAndDirection(A):
	# step 1: Adjust the direction.
	if not geoutils.is_CloseWise(A):
		A.reverse()
	# setp 2: Change the start point based on the MinAreaRectangle.
	OBB,_=geoutils.mininumAreaRectangle(A)
	start_index=OBB.pointTouchRectWithMaxX(A)
	max_X,max_index=A[0][0],0
	for j in range(1,len(A)):
		if A[j][0] > max_X:
			max_X,max_index=A[j][0],j
	distance=OBB.distanceOfPointFromRect(A[max_index])
	if (distance<0.05):
		start_index=max_index
	B=[]
	for i in range(max_index, max_index + len(A)):
		if i == len(A) - 1:
			continue
		B.append(A[i % len(A)])
	# setp 3: Closure the curve.
	B.append(B[0])
	return B

# [x, y, s, n]=[[x1,x2,...,xn],[y1,y2,...,yn],[s1,s2,...,sn],n,L]
# Args:
#   X=[x1,x2,...,xn]: Coordinate x series.
#   S=[s1,s2,...,sn]: Coordinate s series. s ranges from [0,period_size] 
#         coord_size: The number of coordinate X Y S.
#        series_size: The number of expansion series.
def fourier_Series(X,S,series_size):
	coord_size=len(X)
	if (coord_size <= 2 or
		coord_size != len(S)):
		print('ILLEGAL_ARGUMENT')
		return [],[],[],[]
	period_size=S[coord_size-1]
	#print(period_size)
	XA,XB=[],[]
	for i in range(0,series_size):
		XAi,XBi=0,0
		for j in range(0, coord_size-1):
			# define the convolutional function of X(s) to obtain XA,XB.
			fax=lambda s:(X[j]+(X[j+1]-X[j])*(s-S[j])/(S[j+1]-S[j]))*math.cos(i*2*math.pi*s/period_size)*2/period_size
			if S[j] == S[j+1]:
				print(integrate.quad(fax, S[j], S[j+1]))
				exit()
			XAi+=integrate.quad(fax, S[j], S[j+1])[0]
			fbx=lambda s:(X[j]+(X[j+1]-X[j])*(s-S[j])/(S[j+1]-S[j]))*math.sin(i*2*math.pi*s/period_size)*2/period_size
			XBi+=integrate.quad(fbx, S[j], S[j+1])[0]
		XA.append(XAi)
		XB.append(XBi)
	return XA,XB

# S=XA,XB,YA,YB
def constructObjectByFourierSeries(S,series_size,period_size=2*math.pi):
	arc_length=numpy.linspace(0,period_size,40*5-1)
	constructedCoords=[]
	for i in arc_length:
		coord_x,coord_y=S[0]/2,S[2*series_size]/2
		for j in range(1, series_size):
			coord_x+=S[j]*math.cos(j*i*2*math.pi/period_size)+S[series_size+j]*math.sin(j*i*2*math.pi/period_size)
			coord_y+=S[2*series_size+j]*math.cos(j*i*2*math.pi/period_size)+S[3*series_size+j]*math.sin(j*i*2*math.pi/period_size)
		constructedCoords.append([coord_x,coord_y])
	return constructedCoords

# Define some interfaces that allow  other applications to use the Fourier transform
# Args:
#   A=[(x1,y1),(x2,y2),...,(xn,yn)]: The input geometry object.
#                       is_adjusted: The process is used to adjust the start point of object
#                                    for the task of shape description.
#                     is_noemalized: Forces the perimeter to 2*PI.
#                                    This parametry is not necessary after the geometry transform.
# Output:
#   return the vectors of X,Yand S
def convertGeoObjectToArcLengthFunction(A,is_adjusted=False,is_normalized=False):
	X,Y,S=[],[],[]
	coord_size=len(A)
	if (coord_size < 2):
		print('ILLEGAL_ARGUMENT')
		return [],[],[];
	if (A[0][0]!=A[len(A)-1][0] or A[0][1]!=A[len(A)-1][1]):
		A.append(A[0])
		coord_size=coord_size+1
	# Adjust the start point and direction.
	if (is_adjusted):
		A=adjust_StartAndDirection(A)
	for i in range(0,coord_size):
		X.append(A[i][0])
		Y.append(A[i][1])
		if (i==0):
			S.append(0)
		else:
			S.append(S[i-1]+
				math.sqrt(
					pow(A[i][0]-A[i-1][0],2)+
					pow(A[i][1]-A[i-1][1],2))
				)
	# Forces the perimeter to 2*PI.
	if (is_normalized):
		S=[i*2*math.pi/S[len(S)-1] for i in S]
	return X,Y,S

# Define some interfaces that allow  other applications to use the Fourier transform
# Fourier expansion for a geometry object.
# Args:
#   A=[(x1,y1),(x2,y2),...,(xn,yn)]: The input geometry object.
#                       series_size: The number of expansion series.
#                       is_adjusted: The process is used to adjust the start point of object
#                                    for the task of shape description.
#                     is_noemalized: Forces the perimeter to 2*PI.
#                                    This parametry is not necessary after the geometry transform.
# output:
#   XA,XB,YA,YB: The coefficients of Fourier series.
def do_FFT(A,series_size,is_adjusted=True,is_normalized=False):
	X, Y, S = convertGeoObjectToArcLengthFunction(A,is_adjusted=is_adjusted,is_normalized=is_normalized)
	coord_size = len(X)
	if(coord_size == 0):
		print('ILLEGAL_ARGUMENT')
		return [], [], [], []
	XA, XB = fourier_Series(X, S, series_size)
	YA, YB = fourier_Series(Y, S, series_size)
	return XA, XB, YA, YB

# Define some interfaces that allow  other applications to use the Fourier transform
# Args:
#   A=[(x1,y1),(x2,y2),...,(xn,yn)]: The first geometry object.
#   B=[(x1,y1),(x2,y2),...,(xn,yn)]: The second geometry object.
# Output:
#   return the distance bewteen the object A and B.
def do_Matching(A,B,series_size=4):

	# Calculate the basic geometric parameters
	[[ACX,ACY],Aarea,Aperi]=geoutils.get_basic_parametries_of_Poly(A)
	[[BCX,BCY],Barea,Bperi]=geoutils.get_basic_parametries_of_Poly(B)

	#uniform_A_coords = [[(j[0]-ACX)/math.sqrt(Aarea), (j[1]-ACY)/math.sqrt(Aarea)] for j in A]
	#uniform_B_coords = [[(j[0]-BCX)/math.sqrt(Barea), (j[1]-BCY)/math.sqrt(Barea)] for j in B]

	uniform_A_coords = [[(j[0]-ACX)*2*math.pi/Aperi, (j[1]-ACY)*2*math.pi/Aperi] for j in A]
	uniform_B_coords = [[(j[0]-BCX)*2*math.pi/Bperi, (j[1]-BCY)*2*math.pi/Bperi] for j in B]
	
	AXA,AXB,AYA,AYB  = do_FFT(uniform_A_coords,series_size,is_adjusted=True)
	composite_A_vector = AXA+AXB+AYA+AYB
	composite_A_vector = np.array(composite_A_vector)
	# print(composite_A_vector)
	# print(composite_A_vector / AXA[0])
	
	BXA,BXB,BYA,BYB  = do_FFT(uniform_B_coords,series_size,is_adjusted=True)
	composite_B_vector = BXA+BXB+BYA+BYB
	composite_B_vector = np.array(composite_B_vector)
	# print(composite_B_vector)
	# print(composite_B_vector / BXA[0])

	# exit()
	dis = 0
	for i in range(len(composite_A_vector)):
		dis += pow((composite_A_vector[i]-composite_B_vector[i]), 2)
	return dis # get_distance(composite_A_vector,composite_B_vector,1)

def main(argv=None):
	coords = []
	file=open('./data/FF_test.json','r',encoding='utf-8')
	data=json.load(file)
	feature_size=len(data['features'])
	for i in range(0,feature_size):
		ID = data['features'][i]['attributes']['type']        # nCohesion
		geome_dict = data['features'][i]['geometry']          # Get the geometry objects.
		geo_path   = geome_dict['rings']
		coord = []
		for j in range(0,len(geo_path)):
			# print(len(geo_path[j]))
			for k in range(0,len(geo_path[j])):
				coord.append([geo_path[j][k][0], geo_path[j][k][1]])
			break
		coords.append([coord, ID])

	for i in range(0, len(coords)):
		for j in range(i+1, len(coords)):
			print('matching_degres of {0} and {1} is :   {2}'.format(coords[i][1], coords[j][1], do_Matching(coords[i][0], coords[j][0])))
			# print('')
	exit()

if __name__ == '__main__':
	main()



# Test code.
def main_(argv=None):
	'''
	testA=[[1,1],[2,1],[2,2],[1,2],[1,1]]
	print(support.get_basic_parametries_of_Poly(testA))
	return

	X,Y,S=[0,1,1,0,0],[0,0,1,1,0],[0,1,2,3,4,5]
	#print(X,Y,S)
	XA,XB=fourier_Series(X,S,5,5)
	YA,YB=fourier_Series(Y,S,5,5)
	print(XA,XB,YA,YB)

	return'''
	X,Y,S=[],[],[]
	file=open('./data/F_test.json','r',encoding='utf-8')
	data=json.load(file)
	feature_size=len(data['features'])
	for i in range(0,feature_size):
		attri_dict=data['features'][i]['attributes']        # Get the attributes.
		geome_dict=data['features'][i]['geometry']          # Get the geometry objects.
		geo_path=geome_dict['rings']
		for j in range(0,len(geo_path)):
			print(len(geo_path[j]))
			for k in range(0,len(geo_path[j])):
				X.append(geo_path[j][k][0])
				Y.append(geo_path[j][k][1])
				if k==0:
					S.append(0)
				else:
					S.append(S[k-1]+
						math.sqrt(
							pow(geo_path[j][k][0]-geo_path[j][k-1][0],2)+
							pow(geo_path[j][k][1]-geo_path[j][k-1][1],2)
							)
						)
			print(len(X),len(Y),len(S))
			break
		break

	print(len(data['features']))
	#return
	# The coordinates of a polygon maybe organized in a way like:
	# [(x1,y1),(x2,y2),...,(xn,yn)],
	# and it can be converted to [x1,x2,...,xn],[y1,y2,...,yn] as a function of 
	# the arc-length [s1,s2,...,sn],
	# so, the algorithm begins.
	# X,Y,S=[0,1,1,0,0],[0,0,1,1,0],[0,1,2,3,4]
	series_size = 4
	coord_size  = len(X)
	period_size = S[coord_size-1]
	print(coord_size,period_size)
	XA,XB = fourier_Series(X,S,series_size)
	YA,YB = fourier_Series(Y,S,series_size)

	arc_length = numpy.linspace(0,period_size,100*5-1)
	origin_coords_x,origin_coords_y,transform_coords_x,transform_coords_y = [],[],[],[]
	for i in arc_length:
		origin_coord_x,transfor_coord_x = piecewise_function(S,X,i),XA[0]/2
		origin_coord_y,transfor_coord_y = piecewise_function(S,Y,i),YA[0]/2
		for j in range(1, series_size):
			transfor_coord_x += XA[j]*math.cos(j*i*2*math.pi/period_size) + XB[j]*math.sin(j*i*2*math.pi/period_size)
			transfor_coord_y += YA[j]*math.cos(j*i*2*math.pi/period_size) + YB[j]*math.sin(j*i*2*math.pi/period_size)
		origin_coords_x.append(origin_coord_x)
		origin_coords_y.append(origin_coord_y)
		transform_coords_x.append(transfor_coord_x)
		transform_coords_y.append(transfor_coord_y)
	#print(XA)
	#print(XB)
	'''
	plt.subplot(311)
	plt.plot(arc_length,origin_coords_x,color='blue',label='Origin coordinate Y')
	plt.plot(arc_length,transform_coords_x,color='red',label='Fourier series approximation')
	plt.subplot(312)
	plt.plot(arc_length,origin_coords_y,color='blue',label='Origin coordinate Y')
	plt.plot(arc_length,transform_coords_y,color='red',label='Fourier series approximation')
	plt.subplot(313)
	'''
	plt.plot(X, Y, color='blue',marker='o')
	plt.plot(transform_coords_x, transform_coords_y, color='red',marker='o')
	plt.yticks(Y)
	plt.legend()
	plt.gca().set_aspect(1)
	plt.show()
