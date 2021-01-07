# 
# Smallest enclosing circle - Library (Python)
# 
# Copyright (c) 2017 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING.txt and COPYING.LESSER.txt).
# If not, see <http://www.gnu.org/licenses/>.
# 

import math, random
import numpy as np


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
# 
# Initially: No boundary points known
def make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)
    
    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[ : i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[ : i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q
    
    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue
        
        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
            right = c
    
    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox; ay -= oy
    bx -= ox; by -= oy
    cx -= ox; cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0    
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

# This implementation referred to the website:
# https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def find_hull_intersection(equations, ray_point):
    # normalise ray_point
    unit_ray = normalize(ray_point)
    # find the closest line/plane/hyperplane in the hull:
    closest_plane = None
    closest_plane_distance = 0
    for plane in equations:
        normal = plane[:-1]
        distance = plane[-1]
        # if plane passes through the origin then return the origin
        if distance == 0:
            return np.multiply(ray_point, 0) # return n-dimensional zero vector 
        # if distance is negative then flip the sign of both the
        # normal and the distance:       
        if distance < 0:
            np.multiply(normal, -1);
            distance = distance * -1
        # find out how much we move along the plane normal for
        # every unit distance along the ray normal:
        dot_product = np.dot(normal, unit_ray)
        # check the dot product is positive, if not then the
        # plane is in the opposite direction to the rayL
        if dot_product > 0:  
            # calculate the distance of the plane
            # along the ray normal:          
            ray_distance = distance / dot_product
            # is this the closest so far:
            if closest_plane is None or ray_distance < closest_plane_distance:
                closest_plane = plane
                closest_plane_distance = ray_distance
    print(closest_plane_distance)
    # was there no valid plane? (should never happen):
    if closest_plane is None:
        return None
    # return the point along the unit_ray of the closest plane,
    # which will be the intersection point
    return np.multiply(unit_ray, closest_plane_distance)

def get_equation(point1, point2):
    '''equation of plane line'''
    a = point1[1] - point2[1]
    b = point2[0] - point1[0]
    c = point1[0] *point2[1] - point2[0] * point1[1]
    return a, b, c

# Note: the function need to be further improved and tested
def inSegment(p,  point1, point2, point3, point4):
    # line1: point1--point2    line2: point3-point4
    # check if the intersection is on line2 (not including endpoints), and two lines are not parallel.
    # check if p is on line2, and only check the x value, which is p[0].
    if point1[0] == point2[0]:    # if line1 is vertical
        if  p[1] > min(point1[1], point2[1]) and p[1] < max(point1[1], point2[1]):
            if p[0] >= min(point3[0], point4[0]) and p[0] <= max(point3[0], point4[0]):
                return True
    elif point1[1] == point2[1]:  # if line1 is horizontal
        if p[0] > min(point1[0], point2[0]) and p[0] < max(point1[0], point2[0]):
            if p[1] >= min(point3[1], point4[1]) and p[1] <= max(point3[1], point4[1]):
                return True
    else:                         # line 2 is diagonal
        if p[0] > min(point1[0], point2[0]) and p[0] < max(point1[0], point2[0]):
            if point3[0] == point4[0]:    # if line 2 is vertical
                if p[1] >= min(point3[1], point4[1]) and p[1] <= max(point3[1], point4[1]):
                    return True
            elif point3[1] == point4[1]:  # if line 2 is horizontal
                if p[0] >= min(point3[0], point4[0]) and p[0] <= max(point3[0], point4[0]):
                    return True
            if p[1] >= min(point3[1], point4[1]) and p[1] <= max(point3[1], point4[1]) and p[0] >= min(point3[0], point4[0]) and p[0] <= max(point3[0], point4[0]):
                return True
    return False

# This implementation referred to the website:
# https://www.jianshu.com/p/d3c5d600cdab
def get_crossing1(point1, point2, point3, point4):
    a1, b1, c1 = get_equation(point1, point2)
    a2, b2, c2 = get_equation(point3, point4)
    d = a1* b2 - a2 * b1
    p = [0, 0]
    if d == 0: #line1 is parallel to line2.
        return None
    else:
        p[0] = (b1 * c2 - b2 * c1)*1.0 / d
        p[1] = (c1 * a2 - c2 * a1)*1.0 / d

    # if inSegment(p, line1, line2) and getLineLength(p, line1[0]) < 1e-3 and getLineLength(p,line1[1]) < 1e-3:
    # print(math.sqrt(math.pow(p[0]-point3[0],2)+math.pow(p[1]-point3[1],2)) < 1e-10)
    # print(math.sqrt(math.pow(p[0]-point4[0],2)+math.pow(p[1]-point4[1],2)) < 1e-10)
    if inSegment(p, point1, point2, point3, point4):
        return [p[0], p[1]]
    else:
        return None

# This implementation referred to the website:
# http://dec3.jlu.edu.cn/webcourse/t000096/graphics/chapter5/01_1.html
def get_crossing2(point1, point2, point3, point4):
    xa, ya = point1[0], point1[1]
    xb, yb = point2[0], point2[1]
    xc, yc = point3[0], point3[1]
    xd, yd = point4[0], point4[1]

    a = np.matrix([[xb-xa,-(xd-xc)], [yb-ya,-(yd-yc)]])
    delta = np.linalg.det(a)

    # no intersection
    if np.fabs(delta) == 0:
        return None

    # calculate the parameter lambda and miu
    c = np.matrix([[xc-xa,-(xd-xc)], [yc-ya,-(yd-yc)]])
    d = np.matrix([[xb-xa,xc-xa], [yb-ya,yc-ya]])

    lamb = np.linalg.det(c)/delta
    miu = np.linalg.det(d)/delta

    # intersection
    if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
        x = xc + miu*(xd-xc)
        y = yc + miu*(yd-yc)
        return [x, y]
    # intersection is on the extension.
    else:
        return None

def find_intersection(vertices, ray_point):
    # find the closest intersection point in the polygon:
    closest_intersection = None
    closest_distance = 0
    vertices_size = len(vertices)

    is_test = False
    for i in range(0, vertices_size-1):
        # get the intersetion point
        # p = get_crossing1([0, 0], ray_point, vertices[i], vertices[i+1])  # algorithm 1: wrose, inSegment condition
        q = get_crossing2([0, 0], ray_point, vertices[i], vertices[i+1])  # algorithm 2: better
        '''
        # compare two algorithms
        if (p is not None and q is None) or (p is None and q is not None):
            print("i={}    p={}    q={}".format(i, p, q))
            is_test = True
            print("The crossing algorithm should be further tested: type 1")
            # return p if p is not None else q, True
        if p is not None and q is not None:
            if (p[0] - q[0]) > 1e-10 or (p[1] - q[1]) > 1e-10:
                print("p={}    q={}".format(p, q))
                print("The crossing algorithm should be further tested: type 2")
        '''
        # there is a intersection point between these two lines.
        if q is not None:
            ray_distance = math.sqrt(q[0]*q[0] + q[1]*q[1])
            # is this the closest so far:
            if closest_intersection is None or ray_distance > closest_distance:
                closest_intersection = q
                closest_distance = ray_distance
    # return the point along the unit_ray of the closest polygon,
    # which will be the intersection point
    return closest_intersection, is_test



## The Best Fit Ellipse algorithm was borrowed from xxx written by:
from numpy.linalg import eig, inv

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    print(down1)
    res1=np.sqrt(up/down1)

    res2=np.sqrt(up/down2)
    return np.array([res1, res2])