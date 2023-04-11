import numpy as np
import cv2
import time

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2 + fy**2)
    print(v)

    hsv = np.zeros((h,w,3), np.uint8)
    hsv[..., 0 ] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr



cap = cv2.VideoCapture(0)

success, prev = cap.read()

prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)




while 1:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)    
    prevgray = gray


    end = time.time()
    fps =  1 / (end - start)


    flowimg = draw_flow(gray, flow)

    flowimg = cv2.putText(flowimg, "%.2f FPS"%(fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.imshow("flow", flowimg)
    cv2.imshow("HSV FLOW", draw_hsv(flow))

    key = cv2.waitKey(5)
    if key == ord('q'):
        break