
import cv2
import os

image_folder = 'robot'
video_name = 'robot_with_legend_final.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))

for i in range(len(images)):
    video.write(cv2.imread(os.path.join(image_folder, str(i) + ".png")))

cv2.destroyAllWindows()
video.release()
