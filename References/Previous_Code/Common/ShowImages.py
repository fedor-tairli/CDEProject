from PIL import Image
import os
import time
import cv2

base_dir = "/remote/tychodata/ftairli/work/Projects/Temp/Graphs"
sub_dirs = ["Axis", "Core", "Energy", "Xmax"]

while True:
    for sub_dir in sub_dirs:
        preName  = f'Model_4_1 {sub_dir} EpochN '
        for epoch in range(50):  # Replace 1000 with the maximum number of epochs
            image_path = os.path.join(base_dir, sub_dir, f"{preName}{epoch}.png")
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                cv2.imshow('Image', image)
                cv2.waitKey(1000)  # Display the image for 1000 ms (1 second)
                cv2.destroyAllWindows()