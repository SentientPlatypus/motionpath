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

ret, old_frame = cap.read()
center = get_center_img_pipeline(old_frame)
old_frame = np.zeros_like(old_frame)
cv.circle(old_frame, center, 5, (0, 255, 0), -1)


centersLog = [center]

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    blank = np.zeros_like(frame)
    newcenter = get_center_img_pipeline(frame)

    cv.circle(frame, newcenter, 5, (0, 255, 0), -1)
    
    centersLog.append(newcenter)

    for i in range(len(centersLog) - 1):
        cv.line(frame, centersLog[i], centersLog[i+1], (255, 255, 255), 2)
        cv.line(blank, centersLog[i], centersLog[i+1], (255, 255, 255), 2)

    cv.imshow("f", frame)
    cv.imshow("blank", blank)
    cv.waitKey(1)

    
cv.destroyAllWindows()