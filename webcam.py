import cv2
import numpy as np



BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

aruco = cv2.aruco
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco.drawMarker(aruco_dict, 2, 700)
print(aruco_dict)
# second parameter is id number
# last parameter is total image size

cameraMatrix = np.load("./calib_mtx.npy")
distCoeffs = np.load("./calib_dist.npy")
rvecs = np.load("./calib_rvecs.npy")
tvecs = np.load("./calib_tvecs.npy")
 
cap = cv2.VideoCapture(0)

def unproject(polygon):
    rect = np.array([[1,0],
                     [0,0],
                     [0,1],
                     [1,1]])
    r = [np.diag(-rect[:,0]),
         np.diag(-rect[:,1]),
         np.identity(4)]

    A = []
    c = []
    for index,vertice in enumerate(polygon):
        A.append([vertice[0],vertice[1],1,0,0,0,0,0,0,r[0][index][0],r[0][index][1],r[0][index][2],r[0][index][3]])
        A.append([0,0,0,vertice[0],vertice[1],1,0,0,0,r[1][index][0],r[1][index][1],r[1][index][2],r[1][index][3]])
        A.append([0,0,0,0,0,0,vertice[0],vertice[1],1,r[2][index][0],r[2][index][1],r[2][index][2],r[2][index][3]])
        c.append([0])
        c.append([0])
        c.append([1])
    A.append([0,0,0, 0,0,0, 0,0,0, 1,0,0,0])
    c.append([1])
    invA = np.linalg.inv(A)
    result = invA@c
    return np.reshape(result[0:9],(3,3))
    
def segment(a,b):
    return np.array([a,b])

def cross_product(a,b):
    return a[0]*b[1] - a[1]*b[0]

def intersects(a,b):
    a0a1 = np.subtract(a[1],a[0])
    a0b0 = np.subtract(b[0],a[0])
    a0b1 = np.subtract(b[1],a[0])

    b0b1 = np.subtract(b[1],b[0])
    b0a0 = np.subtract(a[0],b[0])
    b0a1 = np.subtract(a[1],b[0])

    return cross_product(a0a1,a0b0) * cross_product(a0a1,a0b1) < 0 and cross_product(b0b1,b0a0) * cross_product(b0b1,b0a1) < 0

def inside(point,polygon):
    base = segment(point,np.add(point,[0,10000]))
    count = 0
    for index,vertice in enumerate(polygon):
        nextEdge = polygon[(index+1)%len(polygon)]
        edge = segment(vertice,nextEdge)
        if intersects(base,edge):
            count+=1

    return count%2==1


 
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    aruco.drawDetectedMarkers(gray, corners)
 
    if len(corners) > 0:
        # unproject
        polygon = corners[0][0]
        transform = unproject(polygon)

        # Detect BBox
        height, width, channels = frame.shape
        rmin = height - 1 
        rmax = 0

        cmin = width - 1
        cmax = 0

        for index in range(len(polygon)):
            r = corners[0][0][index][1]
            c = corners[0][0][index][0]

            rmin = int(r if r < rmin else rmin)
            cmin = int(c if c < cmin else cmin)

            rmax = int(r if r > rmax else rmax)
            cmax = int(c if c > cmax else cmax)

        # Draw BBox
        for r in range(rmin,rmax,1):
            for c in range(cmin,cmax,1):
                origin = transform @ np.array([c,r,1])
                if origin[0] > 0 and origin[0] < 1 and origin[1] > 0 and origin[1] < 1:
                    frame[r,c] = frame[int(height*origin[1]),int(width*origin[0])]

        #for c,r in polygon:
        #    origin = transform @ np.array([c,r,1])
        #    frame[int(origin[1]),int(origin[0])] = [255,255,255]



    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

