import cv2
from Hand_Detection import hand_data
import math
import numpy as np
import glob
import PIL
import os
import os.path
from PIL import Image


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# ox, oy = 500,200


class DragImg:
    def __init__(self, path, posOrigin):
        self.path = path
        self.posOrigin = posOrigin
        if self.path.split(".")[-1].lower() == "png":
            self.imgType = "png"
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.imgType = self.path.split(".")[-1].lower()
            self.img = cv2.imread(self.path)
        # self.img=cv2.resize(self.path,(100,100))
        self.size = self.img.shape[:2]

    def updateImage(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size
        if ox < cursor[1] < ox + w and oy < cursor[2] < oy + h:
            self.posOrigin = cursor[1] - w // 2, cursor[2] - h // 2


list_of_images = []
path = "./images/"
f = r"./images"

# resize
for file in os.listdir(f):
    f_img = f + "/" + file
    img = Image.open(f_img)
    img = img.resize((200, 200))
    img.save(f_img)


image_list = glob.glob(f"{path}*")
for x, i in enumerate(image_list):
    list_of_images.append(DragImg(i, [50 + x * 300, 50]))


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                * 255,
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay_image

    return background


def findDistance(p1, p2, img, lmList_all, draw=True):
    x1, y1 = lmList_all[0][p1][1], lmList_all[0][p1][2]
    x2, y2 = lmList_all[0][p2][1], lmList_all[0][p2][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    return img, length


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, lmList_all, bbox = hand_data(img, bdraw=True)
    if len(lmList_all) != 0:
        if lmList_all[0] != "NULL":
            img, l = findDistance(8, 12, img, lmList_all, draw=True)
            if l < 60:
                cursor = lmList_all[0][8]
                for imsObject in list_of_images:
                    imsObject.updateImage(cursor)

    try:
        for imsObject in list_of_images:
            ox, oy = imsObject.posOrigin
            h, w = imsObject.size
            if imsObject.imgType == "png":
                img = overlay_transparent(img, imsObject.img, ox, oy)
            else:
                img[oy : oy + h, ox : ox + w] = imsObject.img

    except:
        pass
    cv2.imshow("VIDEO", img)
    cv2.waitKey(1)
