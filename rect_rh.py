import numpy as np
import cv2

box=np.array(
 [[1225 , 184 ,1246  ,240],
 [1226 , 187 ,1250  ,238],
 [1268 , 186 ,1297 , 241],
 [1269 , 188 ,1297 , 239],
 [1021 , 538 ,1038 ,620],
 [1024 , 542 ,1043 , 619],
 [1499 , 459 ,1528 , 526],
 [1645 , 444 ,1662 , 530],
 [1584 , 455 ,1610 , 532],
 [1652 , 465 ,1676 , 529],
 [1502 , 461 ,1530 , 526],
 [1588 , 468 ,1619 , 531],
 [1456 , 435 ,1474 , 534],
 [1505 , 460 ,1527 , 528],
 [1654 , 470 ,1674 , 531],
 [1590 , 469 ,1621 , 532],
 [1024 , 547 ,1045 , 617],
 [1504 , 463 ,1529 , 524],
 [1504  ,468 ,1527 , 527],
 [1506 , 472 ,1528 , 525]])

def group_rect(box,iou):
    keep_ = []
    if len(box)>1:
        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        while len(area)>1:
            order = area.argsort()[::-1]
            i=order[0]
            xx1=np.maximum(x1[i],x1[order[1:]])
            yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]])
            yy2=np.minimum(y2[i],y2[order[1:]])
            w=np.maximum(0.,xx2-xx1)
            h=np.maximum(0.,yy2-yy1)
            inter=w*h
            ovr=inter/(area[i]+area[order[1:]]-inter)
            inds=np.where(ovr>iou)+np.array([1])
            inds=np.append(inds,0)
            rect=[np.min(x1[order[inds]]),np.min(y1[order[inds]]),np.max(x2[order[inds]]),np.max(y2[order[inds]])]
            x1 = np.delete(x1, order[inds])
            y1 = np.delete(y1, order[inds])
            x2 = np.delete(x2, order[inds])
            y2 = np.delete(y2, order[inds])
            area = np.delete(area, order[inds])
            if len(inds)>1:
                x1 = np.hstack((x1,rect[0]))
                y1 = np.hstack((y1, rect[1]))
                x2 = np.hstack((x2, rect[2]))
                y2 = np.hstack((y2, rect[3]))
                area=np.hstack((area,((rect[2]-rect[0])*(rect[3]-rect[1]))))
            else:
                keep_.append(list(rect))
        else:
            keep_.append(list([x1[0], y1[0], x2[0], y2[0]]))
    return keep_

box=np.array(box)
box=box+np.array([-1000,-50,-1000,-50])
keep=group_rect(box,0.2)

draw=np.zeros([600,700,3],np.uint8)

for x in box:
    pt1,pt2=(x[0],x[1]),(x[2],x[3])
    cv2.rectangle(draw,pt1,pt2,(255,0,0),1)
	
cv2.imshow('',draw)
cv2.waitKey(0)
for x in keep:
    pt1,pt2=(x[0],x[1]),(x[2],x[3])
    cv2.rectangle(draw,pt1,pt2,(0,0,255),1)
	
cv2.imshow('',draw)
cv2.waitKey(0)

cv2.destroyAllWindows()