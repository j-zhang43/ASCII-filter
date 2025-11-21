import os
import cv2
import sys
import numpy as np
import math
import mss

N=8
SHIFT = 1/1000

# reads from a texture file, 8x80 and returns list of 8x8
def process_texture(filename):
    texture_png = cv2.imread(filename)
    processed_textures = []

    for i in range(10):
        processed_textures.append(texture_png[0:N,i*N:(i+1)*N])

    return processed_textures

# takes img and returns filtered img
def process_img(img, textures):
    height, width, channels = img.shape
    H,W = math.ceil(height/N),math.ceil(width/N)
    avg_img = np.zeros((H,W,channels),np.uint8)
    output_img = np.zeros((H*N,W*N,channels),np.uint8)
    
    # find average color in 8x8 and down scale
    for y in range(0,height, N): 
        for x in range(0,width,N):
            x_end = min(x+N, width)
            y_end = min(y+N, height)

            block = img[y:y_end,x:x_end]
            avg_color = np.mean(block,axis=(0,1))
            avg_img[y//N, x//N] = avg_color

    grey_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

    # placed texture on with gray scale
    for y in range(avg_img.shape[0]):
        for x in range(avg_img.shape[1]):
            bucketed_index = int((grey_img[y,x]+SHIFT)/(25.5+SHIFT))

            output_img[y*N:(y+1)*N,x*N:(x+1)*N] = textures[bucketed_index]

            # color
            block = output_img[y*N:(y+1)*N,x*N:(x+1)*N]
            mask = np.all(block == 255, axis=2)
            block[mask] = avg_img[y, x]
            output_img[y*N:(y+1)*N,x*N:(x+1)*N] = block
    
    # # difference of gaussians and sorbol filter
    # dog = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (0,0), 1) -cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (0,0), 1.4)
    # dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    # _ , thresh = cv2.threshold(dog, 140,255,cv2.THRESH_BINARY)

    # cv2.imshow("gaussian and shorbol",dog)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return output_img


def main():
    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("Usage: filter.py [img_path], no path, for video camera")
        return

    processed_textures = process_texture("texture.png")

    if len(sys.argv) == 2:
        img = cv2.imread(sys.argv[1])

        if img is None:
            print("Cannot open image")
            return
        
        processed_img = process_img(img, processed_textures)

        cv2.imwrite("output.png",processed_img)
    else:

        # MULTI = 2
        # with mss.mss() as sct:
        #     monitor = sct.monitors[1]
        #     while True:
        #         sc = np.array(sct.grab(monitor))
        #         img = cv2.cvtColor(sc, cv2.COLOR_BGRA2BGR)
        #         img =  cv2.resize(img,None,fx=MULTI,fy=MULTI)

        #         cv2.imshow("Frame", process_img(img,processed_textures))

        #         if cv2.waitKey(1) == ord('q'):
        #             break


        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
        
        while True:
            ret,frame = cap.read()

            if not ret:
                print("cannot recieve video")

            cv2.imshow("Frame", process_img(frame,processed_textures))

            if cv2.waitKey(1) == ord('q'):
                break

main()