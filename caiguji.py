from shutil import copytree
import argparse
import os
import cv2
import numpy as np
import re
import imutils


# PART 1: CV code
# because we'll compare two methods, we'll find contour twice
# returns None if there is no contour
def canny_and_find_contour(gray):
    img_canny = cv2.Canny(gray,75,200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img_canny = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    if len(cnts) > 0:
        c = cnts[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return approx
    else:
        return None

def cut_to_content(input_path, output_path):
    # load image and downsample for better performance
    img = cv2.imread(input_path, 1)
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = imutils.resize(img, height = 500)

    # save a gray image; because in some cases it works better 
    # if remove bg and threshold is skipped (more info in the original image)
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use grabCut to smartly remove background
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,
        0,  # threshold value, ignored when using cv2.THRESH_OTSU
        255,  # maximum value assigned to pixel values exceeding the threshold
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding type
    
    cnt = canny_and_find_contour(gray)
    cnt_orig = canny_and_find_contour(gray_orig)
    # if the smart way(with remove bg) cannot find a contour or the contour is too small, 
    # use the dumb way(skip removebg and threshold). 
    if cnt is None or cv2.contourArea(cnt) < 50000:
        cnt = cnt_orig
    x, y, w, h = cv2.boundingRect(cnt) 
    # if the contour of the smart way is too close to the edge, 
    # and the dumb way is acceptable(area > 5000), use the dumb way
    if (y+h+10 >= img.shape[0] or x+w+10 >= img.shape[1]) and cv2.contourArea(cnt_orig) > 50000:
        cnt = cnt_orig
        x, y, w, h = cv2.boundingRect(cnt) 
    if cnt is not None:    
        x, y, w, h = int(x*ratio), int(y*ratio), int(w*ratio), int(h*ratio)
        orig = orig[y:y+h, x:x+w]
    orig = imutils.resize(orig, height = 1000)
    cv2.imwrite(output_path, orig)


# PART 2: working with files
parser = argparse.ArgumentParser(prog='caiguji', description='将古籍裁剪到有内容的部分')
parser.add_argument('input', default='input', nargs='?', help='源目录，默认为.\input')
parser.add_argument('output', default='output', nargs='?', help='目标目录，默认为.\output')
args = parser.parse_args()

def process_file(source_file, target_file):
    if source_file.endswith('.png') or source_file.endswith('.tif') or source_file.endswith('.tiff') or source_file.endswith('.jpg') or source_file.endswith('.jpeg'):
        cut_to_content(source_file, target_file)

copytree(args.input, args.output, copy_function=process_file, dirs_exist_ok=True)

