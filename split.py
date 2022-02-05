import cv2
from imutils import paths
import random
import os


def split(category):
    root_path = "./data/raw/" + category
    image_paths = sorted(list(paths.list_images(root_path)))
    random.seed(42)
    random.shuffle(image_paths)
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        if i < len(image_paths) / 0.6:
            save_path = "./data/train/" + category + "/" + str(i) + ".jpg"
        elif i > len(image_paths) / 0.8:
            save_path = "./data/test/" + category + "/" + str(int(i - len(image_paths) / 0.8)) + ".jpg"
        else:
            save_path = "./data/valid/" + category + "/" + str(int(i - len(image_paths) / 0.6)) + ".jpg"
        cv2.imwrite(save_path, img)
        if i % 1000 == 0:
            print(i)


if not os.path.exists("./data/train/Positive/"):
    os.mkdir("./data/train/Positive/")
    os.mkdir("./data/valid/Positive/")
    os.mkdir("./data/test/Positive/")
    os.mkdir("./data/train/Negative/")
    os.mkdir("./data/valid/Negative/")
    os.mkdir("./data/test/Negative/")

split("Positive")
split("Negative")
