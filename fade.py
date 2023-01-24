import cv2
import os
import numpy as np
from glob import glob

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


image_files = glob(os.path.join('images', '*.*'))
frame_width = 1920
frame_height = 1080
shadow_distance = 20
frame = 0
image_list = []

bg = cv2.imread("wall.jpg")
bg = cv2.resize(bg, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

for file in image_files:
    img = cv2.imread(file)

    ratio = (frame_height - 100) / img.shape[0]
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    print("processing image:", file)
    posx = int((frame_width - new_width) / 2)
    posy = int((frame_height - new_height) / 2)

    right_shadow = np.zeros((new_height, shadow_distance, 3), dtype=np.uint8)
    right_shadow[:, :] = (0, 0, 0)
    right_shadow_blurred = cv2.medianBlur(right_shadow, 13)

    lower_shadow = np.zeros((shadow_distance, new_width - shadow_distance, 3), dtype=np.uint8)
    lower_shadow[:, :] = (0, 0, 0)
    lower_shadow_blurred = cv2.medianBlur(lower_shadow, 13)

    for i in range(31):
        alpha = i / 30
        img_fade = paste_image(bg.copy(), img, posx, posy, alpha)
        img_with_right_shadow = paste_image(img_fade, right_shadow_blurred, posx + new_width, posy + shadow_distance, 0.5 * alpha)
        img_with_all_shadows = paste_image(img_with_right_shadow, lower_shadow_blurred, posx + shadow_distance, posy + new_height, 0.5 * alpha)
        name = "img" + str(frame) + ".jpg"
        cv2.imwrite(name, img_with_all_shadows)
        image_list.append(name)
        frame += 1

    name = "img" + str(frame) + ".jpg"
    cv2.imwrite(name, img_fade)
    frame += 1

    for i in range(150):
        image_list.append(name)
        
    for i in range(30):
        alpha = (30 - i) / 30
        img_fade = paste_image(bg.copy(), img, posx, posy, alpha)
        img_with_right_shadow = paste_image(img_fade, right_shadow_blurred, posx + new_width, posy + shadow_distance, 0.5 * alpha)
        img_with_all_shadows = paste_image(img_with_right_shadow, lower_shadow_blurred, posx + shadow_distance, posy + new_height, 0.5 * alpha)
        name = "img" + str(frame) + ".jpg"
        cv2.imwrite(name, img_with_all_shadows)
        image_list.append(name)
        frame += 1

with open('images.txt', 'w') as f:
    for image in image_list:
        f.write("file '{}'\n".format(image))

os.system('ffmpeg -f concat -i images.txt -c:v libx264 -r 30 -pix_fmt yuv420p -vf scale=1920:1080  -y slideshow.mp4')
os.remove('images.txt')
os.system("del img*.*")
