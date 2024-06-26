
import cv2
import numpy as np
import json
import os
import math

max_disparity = 192
initdisp = 10
expand_window_size = 512


def draw_polywithbbox(left_image, disp_image, polys, bboxes, curpoints, curbboxidx):
    left = left_image.copy()
    for i, p in enumerate(curpoints):
        x, y = p[0], p[1]
        cv2.circle(left, (x, y), 5, (0, 0, 255), 2)
        continue
    
    for i, p in enumerate(polys):
        p = np.array(p)
        cv2.fillPoly(left, [p], (255, 255, 0))

    for i, b in enumerate(bboxes):
        x1, y1, x2, y2, _ = b
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(bboxes) == 0:
        left_expand = np.zeros((expand_window_size, expand_window_size, 3), dtype=np.uint8)
    else:
        left_copy = cv2.copyMakeBorder(left_image, 
                            expand_window_size//2,
                            expand_window_size//2,
                            expand_window_size//2,
                            expand_window_size//2,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        x1, y1, x2, y2, d1 = bboxes[curbboxidx[0]]
        left_focus = left_copy[y1+expand_window_size//2:y2+expand_window_size//2, x1+expand_window_size//2:x2+expand_window_size//2, :]
        left_expand = cv2.resize(left_focus, (expand_window_size, expand_window_size), interpolation=cv2.INTER_LINEAR)

        disp_focus = disp_image[y1:y2, x1:x2]
        mask = disp_focus > 0
        x = np.linspace(0, x2-x1-1, x2-x1, dtype=np.int32)
        y = np.linspace(0, y2-y1-1, y2-y1, dtype=np.int32)
        xv, yv = np.meshgrid(x, y)

        if np.sum(mask) == 0:
            pass
        else:
            xv = xv[mask]
            yv = yv[mask]

            for i in range(np.sum(mask)):
                x, y = xv[i], yv[i]
                d = disp_focus[y, x]
                x = int(x / (x2 - x1) * expand_window_size)
                y = int(y / (y2 - y1) * expand_window_size)
                cv2.circle(left_expand, (x, y), 1, (0, 0, 255), 1)
                cv2.putText(left_expand, str(round(d, 0)), (x, y), 1, 1, (0, 0, 255))
        cv2.putText(left_expand, str(round(d1, 4)), (0, 20), 1, 2, (0, 255, 0))

    cv2.imshow('left', left)
    cv2.moveWindow('left', 50, 50)
    cv2.imshow('left_expand', left_expand)
    cv2.moveWindow('left_expand', 720+0, 540)
    cv2.imshow('disp', cv2.applyColorMap(disp.astype(np.uint8), cv2.COLORMAP_JET))
    cv2.moveWindow('disp', 950, 50)


def mouse_event_left(event, x, y, flags, param):
    points, saveflag = param

    if event == cv2.EVENT_FLAG_LBUTTON:  
        saveflag[0] = False
        points.append([x, y])

def mouse_event_expand(event, x, y, flags, param):
    disp, bboxes, cur_bboxidx = param

    if event == cv2.EVENT_FLAG_LBUTTON:  
        if len(bboxes) == 0:
            return
        else:
            
            x1, y1, x2, y2, d1 = bboxes[cur_bboxidx[0]]

            disp_focus = disp[y1:y2, x1:x2]
            mask = disp_focus > 0
            xL = np.linspace(0, x2-x1-1, x2-x1, dtype=np.int32)
            yL = np.linspace(0, y2-y1-1, y2-y1, dtype=np.int32)
            xv, yv = np.meshgrid(xL, yL)

            if np.sum(mask) == 0:
                return
            else:
                xv = xv[mask]
                yv = yv[mask]
                mindist = 7
                cnt = -1

                for i in range(np.sum(mask)):
                    x_, y_ = xv[i], yv[i]
                    x_ = x_ / (x2 - x1) * expand_window_size
                    y_ = y_ / (y2 - y1) * expand_window_size
                    dist = math.sqrt((x_ - x) * (x_ - x) + (y_ - y) * (y_ - y)) 
                    if mindist > dist:
                        cnt = i
                        mindist = dist
                    continue

                if cnt == -1:
                    return
                else:
                    x_, y_ = xv[cnt], yv[cnt]
                    d = disp_focus[y_, x_]
                    bboxes[cur_bboxidx[0]][-1] = float(d)


def keyevent(key, output_name, cur_image_idx, cur_bboxidx, cur_points, polys, bboxes, cur_anno_mode, saveflag):
    if key == -1:
        pass
    else:
        print(key)
        #print(cur_image_idx, cur_bboxidx, cur_points, polys, bboxes, cur_anno_mode, saveflag)
    if key == 105: # i
        with open(output_name, "w") as json_file:
            dict_d = {
                'polys': polys,
                'bboxes': bboxes
            }
            json.dump(dict_d, json_file)

        saveflag[0] = True
        cur_points = []
        return polys, bboxes, cur_points
    elif key == 107: # k
        if cur_anno_mode[0] == 'bbox' and len(bboxes) == 0:
            return polys, bboxes, cur_points
        if cur_anno_mode[0] == 'poly' and len(polys) == 0:
            return polys, bboxes, cur_points
        
        if cur_anno_mode[0] == 'bbox':
            cur_points = []
            bboxes.pop(-1)
            cur_bboxidx[0] = cur_bboxidx[0] - 1
            if len(bboxes) == 0:
                cur_bboxidx[0] = -1
        if cur_anno_mode[0] == 'poly':
            polys.pop(-1)
        return polys, bboxes, cur_points
    elif key == 100: # d
        if cur_anno_mode[0] == 'bbox':
            return polys, bboxes, cur_points
        
        cur_anno_mode[0] = 'bbox'
        cur_points = []

        return polys, bboxes, cur_points
    elif key == 97: # a
        if len(cur_points) > 2:
            polys.append(cur_points)

        cur_points = []

        if cur_anno_mode[0] == 'poly':
            return polys, bboxes, cur_points
        
        cur_anno_mode[0] = 'poly'

        return polys, bboxes, cur_points
    elif key == 119: # w
        cur_points = []
        cur_bboxidx[0] = cur_bboxidx[0] + 1
        if len(bboxes) == 0:
            cur_bboxidx[0] = -1
            return polys, bboxes, cur_points

        if cur_bboxidx[0] == len(bboxes):
            cur_bboxidx[0] = len(bboxes) - 1
        return polys, bboxes, cur_points
    elif key == 112: # p
        cur_points = []
        if saveflag[0] == False:
            return polys, bboxes, cur_points

        cur_image_idx[0] = cur_image_idx[0] + 1
        if cur_image_idx[0] == len(imagelist):
            cur_image_idx[0] = len(imagelist) - 1

        if os.path.isfile('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json'):
            with open('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json', "r") as json_file:
                dict_d = json.load(json_file)
            polys = dict_d['polys']
            bboxes = dict_d['bboxes']
            if len(bboxes) == 0:
                cur_bboxidx[0] = -1
            cur_bboxidx[0] = 0
        else:
            polys = []
            bboxes = []
            cur_bboxidx[0] = -1

        print(cur_image_idx)
        return polys, bboxes, cur_points
    elif key == 115: # s
        cur_points = []
        cur_bboxidx[0] = cur_bboxidx[0] - 1
        if len(bboxes) == 0:
            cur_bboxidx[0] = -1
            return polys, bboxes, cur_points

        if cur_bboxidx[0] == -1:
            cur_bboxidx[0] = 0
        return polys, bboxes, cur_points
    elif key == 111: # o
        cur_points = []
        if saveflag[0] == False:
            return polys, bboxes, cur_points

        cur_image_idx[0] = cur_image_idx[0] - 1
        if cur_image_idx[0] == -1:
            cur_image_idx[0] = 0

        if os.path.isfile('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json'):
            with open('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json', "r") as json_file:
                dict_d = json.load(json_file)
            polys = dict_d['polys']
            bboxes = dict_d['bboxes']
            if len(bboxes) == 0:
                cur_bboxidx[0] = -1
            cur_bboxidx[0] = 0
        else:
            polys = []
            bboxes = []

            cur_bboxidx[0] = -1

        print(cur_image_idx)
        return polys, bboxes, cur_points
    else:
        return polys, bboxes, cur_points
    
    return


if __name__ == '__main__':
    imagelist = os.listdir('./rainy/left-image-half-size/')

    cur_image_idx = [0]
    cur_bboxidx = [-1]
    cur_points = [] # [[x, y], ]
    cur_anno_mode = ['bbox']
    saveflag = [False]
    cv2.namedWindow('left')
    cv2.namedWindow('disp')
    cv2.namedWindow('left_expand')

    if os.path.isfile('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json'):
        with open('./rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json', "r") as json_file:
            dict_d = json.load(json_file)
        polys = dict_d['polys']
        bboxes = dict_d['bboxes']
        if len(bboxes) == 0:
            pass
        else:
            cur_bboxidx[0] = 0
    else:
        polys = [] # [[[x, y], ...], ...]
        bboxes = [] # [[x1, y1, x2, y2], ...]

    while True:
        left_name = './rainy/left-image-half-size/' + imagelist[cur_image_idx[0]]
        disp_name = './rainy/disparity-map-half-size/' + imagelist[cur_image_idx[0]].split('.')[0] + '.png'
        output_name = './rainy/anno/' + imagelist[cur_image_idx[0]].split('.')[0] + '.json'
        
        left = cv2.imread(left_name)
        disp = cv2.imread(disp_name, cv2.IMREAD_UNCHANGED) 
        disp = np.ascontiguousarray(disp, dtype=np.float32) / 256
        draw_polywithbbox(left, disp, polys, bboxes, cur_points, cur_bboxidx)
        
        cv2.setMouseCallback("left", mouse_event_left, [cur_points, saveflag])
        cv2.setMouseCallback("left_expand", mouse_event_expand, [disp, bboxes, cur_bboxidx])
        key = cv2.waitKey(5)

        polys, bboxes, cur_points = keyevent(key, output_name, cur_image_idx, cur_bboxidx, cur_points, polys, bboxes, cur_anno_mode, saveflag)

        if cur_anno_mode[0] == 'bbox' and len(cur_points) == 2:
            if len(bboxes) == 0:
                cur_bboxidx[0] = 0
            else:
                cur_bboxidx[0] = cur_bboxidx[0] + 1

            x1, y1, x2, y2 = cur_points[0][0], cur_points[0][1], cur_points[1][0], cur_points[1][1]
            bboxes.append([x1, y1, x2, y2, -1])
            cur_points = []

        if key == 27: # esc
            break

        continue
