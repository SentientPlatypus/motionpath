import numpy as np
import cv2 as cv


def get_center_img_pipeline(img):
    blur_image = cv.GaussianBlur(img, (5, 5), 0)

    lower_hsv = np.array([71, 18, 0]) # lower threshold for H, S, and V respectively
    upper_hsv = np.array([132, 255, 197])

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, lower_hsv, upper_hsv)

    contours, h = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    moments = cv.moments(largest_contour)

    # Find the centroid of the largest contour
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])

    # Draw the centroid on the original image
    return (cx, cy)



cap = cv.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (100, 100),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000, 0.01))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
center = get_center_img_pipeline(old_frame)
old_frame = np.zeros_like(old_frame)
cv.circle(old_frame, center, 5, (0, 255, 0), -1)


old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    base = frame
    center = get_center_img_pipeline(frame)



    # frame = np.zeros_like(frame)
    cv.circle(frame, center, 5, (0, 255, 0), -1)
    cv.circle(base, center, 5, (0, 255, 0), -1)
    cv.imshow("BASE", base)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()