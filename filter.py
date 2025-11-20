import os
import cv2
import sys
import numpy as np

N=8
SHIFT = 1/1000

# reads from a texture file, 8x80 and returns list of 8x8
def process_texture(filename):
    texture_png = cv2.imread(filename)
    processed_textures = []

    for i in range(10):
        processed_textures.append(texture_png[0:N,i*N:(i+1)*N])

    return processed_textures

# takes img and resize and average 8x8
def process_img(img, textures):
    height, width, channels = img.shape
    avg_img = np.zeros((height//N,width//N,channels),np.uint8)
    output_img = np.zeros((height//N*8,width//N*8,channels),np.uint8)
    
    for y in range(0,height, N):
        for x in range(0,width,N):
            x_end = min(x+N, width)
            y_end = min(y+N, height)

            block = img[y:y_end,x:x_end]
            avg_color = np.mean(block,axis=(0,1))
            avg_img[y//N, x//N] = avg_color

    grey_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

    for y in range(avg_img.shape[0]):
        for x in range(avg_img.shape[1]):
            bucketed_index = int((grey_img[y,x]+SHIFT)/(25.5+SHIFT))
            output_img[y*N:(y+1)*N,x*N:(x+1)*N] = textures[bucketed_index]

    return output_img


def main():
    if len(sys.argv) != 2:
        print("Usage: filter.py [img_path]")
        return

    processed_textures = process_texture("texture.png")
    img = cv2.imread(sys.argv[1])

    if img is None:
        print("Cannot open image")
        return
    
    processed_img = process_img(img, processed_textures)
    cv2.imwrite("output.png",processed_img)

main()

