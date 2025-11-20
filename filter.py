import os
import cv2
import sys


# reads from a texture file, 8x80 and returns list of 8x8
def process_texture(filename):
    texture_png = cv2.imread(filename)
    processed_textures = []

    for i in range(10):
        processed_textures.append(texture_png[0:8,i*8:(i+1)*8])

    return processed_textures

def main():
    if len(sys.argv) != 2:
        print("Usage: filter.py [img_path]")
        return

    processed_textures = process_texture("texture.png")
    img = cv2.imread(sys.argv[1])

    if img is None:
        print("Cannot open image")
        return

    resized = cv2.resize(img,(img.shape[1]//8,img.shape[0]//8))
    greyscale = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("resized", resized)
    cv2.imshow("grey", greyscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()

