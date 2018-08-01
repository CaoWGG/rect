import numpy as np
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
            w=np.maximum(0.,xx2-xx1+1)
            h=np.maximum(0.,yy2-yy1+1)
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

def find_rect(rect,point,pick):
    points=[]
    for p in point:
        x,y=int((p[0]+p[2])/2),((p[1]+p[3])/2)
        for i in rect:
            if i[0]<x and x<(i[0]+i[2]):
                if i[1]<y and y<(i[1]+i[3]):
                    points.append([p[0]-i[0]+pick[i[4]][0],
                                   p[1]-i[1]+pick[i[4]][1],
                                   p[2]-i[0]+pick[i[4]][0],
                                   p[3]-i[1]+pick[i[4]][1]])
    return points

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if y+stepSize> image.shape[0] and x+stepSize> image.shape[1]:
                yield (image.shape[1]-windowSize[1],image.shape[0]-windowSize[0],image[image.shape[0]-windowSize[0]:image.shape[0],image.shape[1]-windowSize[1]:image.shape[1]])
            elif y+stepSize> image.shape[0]:
                yield (x,image.shape[0]-windowSize[0],image[image.shape[0]-windowSize[0]:image.shape[0], x:x + windowSize[0]])
            elif x+stepSize> image.shape[1]:
                yield (image.shape[1]-windowSize[1],y,image[y:y + windowSize[1], image.shape[1]-windowSize[1]:image.shape[1]])
            else:
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def updata(frame_pick,point):
    frame_pick[4]=frame_pick[3]
    frame_pick[3]=frame_pick[2]
    frame_pick[2]=frame_pick[1]
    frame_pick[1]=frame_pick[0]
    frame_pick[0]=point
    return frame_pick

def expend_rect(pick,frame):
    w,h=frame.shape[1]-1,frame.shape[0]-1
    pick=pick+np.array([-20,-20,20,20])
    pick=np.minimum([w,h,w,h],pick)
    pick=np.maximum([0, 0, 0, 0], pick)
    return pick

def get_frame_rect(frame_pick):
    pick = list(frame_pick[4])
    pick.extend(frame_pick[3])
    pick.extend(frame_pick[2])
    pick.extend(frame_pick[1])
    pick.extend(frame_pick[0])
    return pick
