# Most of this code is borrowed from https://github.com/Azmarie/Face-Morphing

import cv2
import numpy as np
import random
import sys
import os
import dlib
from glob import glob
from skimage import io
from imutils import face_utils
import math
from subprocess import Popen, PIPE
from PIL import Image

image_list = []
frame_width = 1920
frame_height = 1080
frame_number = 0
frame_rate = 24
morph_len = 5
posx = 0
posy = 0
bg = None


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def generate_morph_sequence(duration,frame_rate,img1,img2,points1,points2,tri_list,size, frame):
    num_images = int(duration*frame_rate)
    #p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', output], stdin=PIPE)
   
    for j in range(0, num_images):
        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points = []
        alpha = j/(num_images-1)

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        
        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype = img1.dtype)

        for i in range(len(tri_list)):    
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time
            morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
            
            pt1 = (int(t[0][0]), int(t[0][1]))
            pt2 = (int(t[1][0]), int(t[1][1]))
            pt3 = (int(t[2][0]), int(t[2][1]))

            #draw lines
            #cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            #cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            #cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)
            
        name = "img" + str(frame) + ".jpg"
        frame += 1
        res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        res.save("temp.jpg", "JPEG")
        res = cv2.imread("temp.jpg")
        os.remove("temp.jpg")
        res = paste_image(bg, res, posx, posy, 1)
        cv2.imwrite(name,res)
        image_list.append(name)

    #p.stdin.close()
    #p.wait()

# Check if a point is inside a rectangle
def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):

    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

    dictionary1 = {}
    return list4

def make_delaunay(f_w, f_h, theList, img1, img2):

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    theList = theList.tolist()
    points = [(int(x[0]),int(x[1])) for x in theList]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
    # Return the list.
    return list4

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[-diff0:avg0,-diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]

def generate_face_correspondences(theImage1, theImage2):
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    imgList = crop_image(theImage1,theImage2)
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)

        try:
            if len(dets) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        j=j+1

        for k, rect in enumerate(dets):
            
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            # corresp = face_utils.shape_to_np(shape)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            currList.append((1,1))
            currList.append((size[1]-1,1))
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1),(size[0]-1)//2))

    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    
    return [size,imgList[0],imgList[1],list1,list2,narray]


def morph(img1, img2, duration, frame_rate, frame):
	[size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
	tri = make_delaunay(size[1], size[0], list3, img1, img2)
	generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, frame)


def paste_image(background_image, foreground_image, position_x, position_y, alpha):
    # Create a ROI on the background image
    rows, cols, channels = foreground_image.shape
    roi = background_image[position_y:rows+position_y, position_x:cols+position_x]

    # Create a mask of the foreground image and its inverse
    img2gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.merge((mask, mask, mask))

    # Blend the images using the alpha value and the mask
    blended = cv2.addWeighted(roi, 1-alpha, foreground_image, alpha, 0, mask)
   
    # Put blended image in ROI and modify the background image
    background_image[position_y:rows+position_y, position_x:cols+position_x] = blended
    return background_image


if __name__ == "__main__":
    bg = cv2.imread("wall.jpg")
    bg = cv2.resize(bg, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    image_files = glob(os.path.join('images', '*.*'))
    img = cv2.imread(image_files[0])
    ratio = (frame_height - 100) / img.shape[0]
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    for i in range(1, len(image_files) - 1):
        file = image_files[i]
        print("processing image:", file)
        img2 = cv2.imread(file)
        ratio = (frame_height - 100) / img2.shape[0]
        new_width = int(img2.shape[1] * ratio)
        new_height = int(img2.shape[0] * ratio)
        img2 = cv2.resize(img2, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        posx = int((frame_width - new_width) / 2)
        posy = int((frame_height - new_height) / 2)
        morph(img, img2, morph_len, frame_rate, frame_number)
        img = img2
        frame_number += frame_rate * morph_len

    with open('images.txt', 'w') as f:
        for image in image_list:
            f.write("file '{}'\n".format(image))

    os.system('ffmpeg -f concat -i images.txt -c:v libx264 -r 30 -pix_fmt yuv420p -vf scale=1920:1080  -y morpheshow.mp4')
    os.remove('images.txt')
    os.system("del img*.*")