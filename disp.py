
import numba
import time
import cv2
import numpy as np
import torch
import os

max_disparity = 192
initdisp = 10
focus_window_size = 100
expand_window_size = 249


def drawimage(left_image, right_image, points, cur_point_idx):
    #print(points, cur_point_idx[0])
    left = left_image.copy()
    right = right_image.copy()
    for i, p in enumerate(points):
        x, y, d = p[0], p[1], p[2]
        if i == cur_point_idx[0]:
            cv2.circle(left, (x, y), 5, (0, 0, 255), 2)
            cv2.putText(left, str(d), (x, y-3), 1, 1, (0, 0, 255), 1)
            cv2.circle(right, (x-d, y), 5, (0, 0, 255), 2)
        else:
            cv2.circle(left, (x, y), 3, (0, 255, 255), 1)
            cv2.putText(left, str(d), (x, y-3), 1, 1, (0, 255, 255), 1)
            cv2.circle(right, (x-d, y), 3, (0, 255, 255), 1)
        continue

    if len(points) == 0:
        left_expand = np.zeros((expand_window_size, expand_window_size, 3), dtype=np.uint8)
        right_expand = np.zeros((expand_window_size, expand_window_size, 3), dtype=np.uint8)
    else:
        left_copy = cv2.copyMakeBorder(left_image, 
                            expand_window_size//2,
                            expand_window_size//2,
                            expand_window_size//2,
                            expand_window_size//2,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        right_copy = cv2.copyMakeBorder(right_image,
                            expand_window_size//2,
                            expand_window_size//2,
                            expand_window_size//2 + max_disparity,
                            expand_window_size//2,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        x, y, d = points[cur_point_idx[0]]
        x1, y1 = x - focus_window_size//2 + expand_window_size//2, y - focus_window_size//2 + expand_window_size//2
        x2, y2 = x + focus_window_size//2 + expand_window_size//2, y + focus_window_size//2 + expand_window_size//2
        left_focus = left_copy[y1:y2, x1:x2, :]
        right_focus = right_copy[y1:y2, x1 + max_disparity - d:x2 + max_disparity - d, :]

        left_expand = cv2.resize(left_focus, (expand_window_size, expand_window_size), interpolation=cv2.INTER_LINEAR)
        right_expand = cv2.resize(right_focus, (expand_window_size, expand_window_size), interpolation=cv2.INTER_LINEAR)

        color = np.zeros((11, 11, 3), dtype=np.uint8)
        color[:, :, 0] = 255
        x1 = expand_window_size//2 - 5
        x2 = expand_window_size//2 + 6
        dif = np.abs(left_expand[x1:x2, x1:x2, :].astype(np.float32) - right_expand[x1:x2, x1:x2, :].astype(np.float32)).mean()
        left_expand[x1:x2, x1:x2, :] = (left_expand[x1:x2, x1:x2, :].astype(np.float32) / 2 + color.astype(np.float32) / 2).astype(np.uint8)
        right_expand[x1:x2, x1:x2, :] = (right_expand[x1:x2, x1:x2, :].astype(np.float32) / 2 + color.astype(np.float32) / 2).astype(np.uint8)
        cv2.putText(left_expand, str(dif)[:6], (10, 30), 1, 2, (0, 0, 255), 1)

    cv2.imshow('left', left)
    cv2.moveWindow('left',480, 0)
    cv2.imshow('right', right)
    cv2.moveWindow('right', 480, 488)
    cv2.imshow('left_expand', left_expand)
    cv2.moveWindow('left_expand', 480+0, 150+540)
    cv2.imshow('right_expand', right_expand)
    cv2.moveWindow('right_expand', 480+960, 150+540)


def mouse_event(event, x, y, flags, param):
    points, cur_point_idx, saveflag = param

    saveflag[0] = False
    if event == cv2.EVENT_FLAG_LBUTTON:  
        if len(points) == 0:
            points.append([x, y, initdisp])
            cur_point_idx[0] = 0
        else:
            points.append([x, y, initdisp])


if __name__ == '__main__':
    imagelist = os.listdir('./samples/Limg/')

    cur_point_idx = [-1]
    cur_image_idx = 0
    saveflag = [False]
    cv2.namedWindow('left')
    cv2.namedWindow('right')
    cv2.namedWindow('left_expand')
    cv2.namedWindow('right')
    cv2.namedWindow('right_expand')

    if os.path.isfile('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy'):
        points = np.load('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy')
        points = points.tolist()
    else:
        points = []

    while True:
        left_name = './samples/Limg/' + imagelist[cur_image_idx]
        right_name = './samples/Rimg/' + imagelist[cur_image_idx]
        output_name = './samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy'
        
        left = cv2.imread(left_name)
        right = cv2.imread(right_name)
        drawimage(left, right, points, cur_point_idx)
        
        cv2.setMouseCallback("left", mouse_event, [points, cur_point_idx, saveflag])
        key = cv2.waitKey(5)
        
        if key == 105: # i
            if len(points) == 0:
                continue
            points_np = np.array(points)
            np.save(output_name, points_np)
            saveflag[0] = True
            continue
        elif key == 107: # k
            if len(points) == 0:
                continue
            points.pop(-1)
            cur_point_idx[0] = cur_point_idx[0] - 1
            
            if len(points) == 0:
                cur_point_idx[0] = -1
            continue
        elif key == 100: # d
            if len(points) == 0:
                continue
            _, _, d = points[cur_point_idx[0]]
            d = d + 1
            if d < 0:
                d = 0
            if d == max_disparity:
                d = max_disparity - 1
            points[cur_point_idx[0]][2] = d
        elif key == 97: # a
            if len(points) == 0:
                continue
            _, _, d = points[cur_point_idx[0]]
            d = d - 1
            if d < 0:
                d = 0
            if d == max_disparity:
                d = max_disparity - 1
            points[cur_point_idx[0]][2] = d
        elif key == 119: # w
            cur_point_idx[0] = cur_point_idx[0] + 1
            if len(points) == 0:
                cur_point_idx[0] = -1
                continue

            if cur_point_idx[0] == len(points):
                cur_point_idx[0] = len(points) - 1
            continue
        elif key == 112: # p
            if saveflag == False:
                continue

            cur_image_idx = cur_image_idx + 1
            if cur_image_idx == len(imagelist):
                cur_image_idx = len(imagelist) - 1

            if os.path.isfile('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy'):
                points = np.load('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy')
                points = points.tolist()
                cur_point_idx[0] = 0
            else:
                points = []
                cur_point_idx[0] = -1

            print(cur_image_idx)
            continue
        elif key == 115: # s
            cur_point_idx[0] = cur_point_idx[0] - 1
            if len(points) == 0:
                cur_point_idx[0] = -1
                continue

            if cur_point_idx[0] == -1:
                cur_point_idx[0] = 0
            continue
        elif key == 111: # o
            if saveflag == False:
                continue

            cur_image_idx = cur_image_idx - 1
            if cur_image_idx == -1:
                cur_image_idx = 0

            if os.path.isfile('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy'):
                points = np.load('./samples/anno/' + imagelist[cur_image_idx].split('.')[0] + '.npy')
                points = points.tolist()
                cur_point_idx[0] = 0
            else:
                points = []
                cur_point_idx[0] = -1

            print(cur_image_idx)
            continue
        elif key == 27:
            break

        continue
