import numpy as np
from numpy.linalg import norm
from shapely.plotting import plot_polygon
from shapely import Point, Polygon
import matplotlib.pyplot as plt
import re



def point_is_inside_polygon(point: Point, polygon: Polygon) -> bool:
    """function to check if a point lies inside a polygon

    Args:
        point (Point): point to check
        polygon (Polygon): polygon being checked against

    Returns:
        bool: returns true if point is inside polygon; returns false if point is outside polygon
    """  
    point = Point(point)  
    polygon = Polygon(polygon)
    print('point is inside polygon: ', point.within(polygon))
    return point.within(polygon)



def polygon_is_valid(vertices: np.array) -> bool:
    """function to check if a polygon is valid

    Args:
        vertices (np.array): polygon vertices being checked

    Returns:
        bool: returns true if polygon is a valid polygon; returns false if it is an invalid polygon (has less than 3 vertices or self-intersects)
    """    
    n = len(vertices)
    if n < 3:
        print('valid: False')
        return False
    
    # loop to iterate over each vertex of the polygon
    for i in range(n):
        # variable for the current vertex in the loop
        p1 = vertices[i]
        # variable for the next vertex in the loop
        p2 = vertices[(i + 1) % n]
        # nested loop to iterate over the remaining vertices in the polygon
        for j in range(i + 2, n + i - 1):
            # variable for the current vertex in the nested loop
            p3 = vertices[j % n]
            # variable for the next vertex in the nested loop
            p4 = vertices[(j + 1) % n]
            # check if current edge intersects with another edge in the polygon
            if intersect(p1, p2, p3, p4):
                print('valid: False')
                return False
    print('valid: True')
    return True



def intersect(p1: np.array, p2: np.array, p3: np.array, p4: np.array) -> bool:
    """function to check if the lines intersect

    Args:
        p1 (np.array): first point checked
        p2 (np.array): second point checked
        p3 (np.array): third point checked
        p4 (np.array): fourth point checked

    Returns:
        bool: returns true if the lines intersect; returns false if they do not intersect
    """        
    # calculates the determinant of a matrix formed by two vectors, which is used to check if the two lines formed by the points intersect
    d = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    # if lines don't intersect, return false
    if d == 0:
        return False
    # check if the lines intersect
    u = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / d
    v = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
    # returns true if the lines intersect (if both 'u' and 'v' are between 0 and 1)
    return (u >= 0) and (v >= 0) and (u <= 1) and (v <= 1)



def create_module(l: float, w: float, a_hat: np.array, pos: np.array) -> Polygon:
    """function that generates rectangular polygon with a given orientation

    Args:
        l (float): length of polygon
        w (float): width of polygon
        a_hat (np.array): alignment vector
        pos (np.array): starting position of the polygon (its bottom left corner)

    Returns:
        Polygon: returns a polygon created using the entered length, width, alignment, and starting position
    """    
    rotated_a_hat = np.array([-a_hat[1], a_hat[0]])
    a_hat_string = str(a_hat)
    a_hat_split_string = re.findall('[0-9]', a_hat_string)
    a_hat_split = (int(a_hat_split_string[0]), int(a_hat_split_string[1]))
    bottom_left = np.array(pos)
    # if a_hat is horizontal (on x-axis (1,0))
    if (a_hat_split[1] == 0 and a_hat_split[0] != 0):
        bottom_right = pos + np.array(a_hat) * w
        top_left = pos + np.array(rotated_a_hat) * l
        top_right = pos + (np.array(rotated_a_hat) * l) + (np.array(a_hat) * w)
     # if a_hat is vertical (on y-axis (0,1))
    if (a_hat_split[0] == 0 and a_hat_split[1] != 0):
        bottom_right = pos + np.array(rotated_a_hat) * l
        top_left = pos + np.array(a_hat) * w
        top_right = pos + (np.array(a_hat) * w) + (np.array(rotated_a_hat) * l)
    # if a_hat is diagonal
    if (a_hat_split[0] != 0 and a_hat_split[1] != 0):
        bottom_right = pos + np.array(a_hat) * w
        top_right = bottom_right + np.array(rotated_a_hat) * l
        top_left = pos + np.array(rotated_a_hat) * l
    # add each vertex to the module
    module_list = [bottom_left.tolist(), bottom_right.tolist(), top_right.tolist(), top_left.tolist()]
    module = Polygon(module_list)
    print(module)
    return module



def poly1_inside_poly2(p1: Polygon, p2: Polygon) -> bool:
    """function to check if a polygon (p1) is inside of another polygon (p2)

    Args:
        p1 (Polygon): inside polygon
        p2 (Polygon): outside polygon

    Returns:
        bool: returns true if p1 is inside p2
    """    
    p1 = Polygon(p1)
    p2 = Polygon(p2)
    p1_points = p1.exterior.coords[:]
    for vertex in p1_points:
        if not point_is_inside_polygon(vertex, p2):
            print('poly1 inside poly2: False')
            return False
    print('poly1 inside poly2: True')
    return True



def rotate_polygon(a_hat: np.array, vector: np.array, polygon: Polygon) -> Polygon:
    """function to rotate the polygon (R) based on the angle between the alignment vector and another vector (x or y axis)

    Args:
        polygon (Polygon): polygon (R) to be rotated

    Returns:
        Polygon: returns a rotated polygon
    """        
    dot_product = np.dot(a_hat, vector)
    cosine = dot_product/(norm(a_hat)*norm(vector))
    rotated_polygon = []
    print('dot product:', dot_product)
    print('cosine:', cosine)
    polygon = Polygon(polygon)
    points = polygon.exterior.coords[:]
    for point in points:
        xr = (point[0]*np.cos(cosine)) - (point[1]*np.sin(cosine))
        yr = (point[0]*np.sin(cosine)) + (point[1]*np.cos(cosine))
        rot_matrix = (xr, yr)
        print('rotated matrix:', rot_matrix)
        rotated_polygon.append(rot_matrix)
    print('rotated polygon:', rotated_polygon)
    return rotated_polygon



def reverse_rotate_polygon(a_hat: np.array, vector: np.array, polygon: Polygon) -> Polygon:
    """rotates polygon (basic bounding box) to original orientation (of polygon (R) inside bounding box)

    Args:
        polygon (Polygon): polygon to be rotated back (basic bounding box)

    Returns:
        Polygon: returns a polygon (bounding box) oriented along alignment vector
    """        
    dot_product = np.dot(a_hat, vector)
    cosine = -(dot_product/(norm(a_hat)*norm(vector)))
    reverse_rotated_polygon = []
    print('dot product:', dot_product)
    print('cosine:', cosine)
    polygon = Polygon(polygon)
    points = polygon.exterior.coords[:]
    for point in points:
        xr = (point[0]*np.cos(cosine)) - (point[1]*np.sin(cosine))
        yr = (point[0]*np.sin(cosine)) + (point[1]*np.cos(cosine))
        rot_matrix = (xr, yr)
        print('reverse rotated matrix:', rot_matrix)
        reverse_rotated_polygon.append(rot_matrix)
    print('reverse rotated polygon:', reverse_rotated_polygon)
    return reverse_rotated_polygon



def find_bbox_basic(points: Polygon) -> Polygon:
    """function to find the basic bounding box with either x or y axis alignment

    Args:
        points (np.array): polygon points used to create the bounding box

    Returns:
        Polygon: returns a basic bounding box for rotated polygon (rotated R)
    """    
    x_coords, y_coords = zip(*points)
    basic_case_bbox = Polygon([(min(x_coords), max(y_coords)), (max(x_coords), max(y_coords)), (max(x_coords), min(y_coords)), (min(x_coords), min(y_coords))])
    print(basic_case_bbox)
    return basic_case_bbox



def find_bbox(a_hat: np.array, vector: np.array, polygon: Polygon) -> Polygon:
    """function to find the smallest box (bounding box) to contain polygon (R) based on alignment vector

    Args:
        a_hat (np.array): alignment vector
        R (Polygon): polygon to be bounded

    Returns:
        Polygon: returns a bounding box for polygon (R)
    """
    a_hat = np.array(a_hat)
    vector = np.array(vector)
    polygon = Polygon(polygon)
    rotated_R = rotate_polygon(a_hat, vector, polygon)
    basic_bbox = find_bbox_basic(rotated_R)
    bbox = Polygon(reverse_rotate_polygon(a_hat, vector, basic_bbox))
    print('bbox:', bbox)
    return bbox


def run_test_cases(func, *args, **kwargs):
    polygon = func(*args, **kwargs)
    plot_polygon(polygon)
    return plt.show()

    

a_hat = [1, 1]
vector = [1, 0]
R = [(0, 2.5), (0.5, 2), (1, 1.75), (1.5, 1.5), (2, 1), (1.5, 3)]
bbox = find_bbox(a_hat, vector, R)
run_test_cases(find_bbox, a_hat, vector, R)

l = 2
w = 1
pos = [1.5, 1.5]
module1 = create_module(l, w, a_hat, pos)
run_test_cases(create_module, l, w, a_hat, pos)
poly1_inside_poly2(module1, R)

l = 0.5
w = 0.25
pos = [1.25, 2]
module2 = create_module(l, w, a_hat, pos)
run_test_cases(create_module, l, w, a_hat, pos)
poly1_inside_poly2(module2, R)
