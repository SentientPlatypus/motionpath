import numpy as np
import cv2 as cv



cap = cv.VideoCapture("nerf.mp4")


centersLog = []
framecount = 0
while(1):
    ret, img = cap.read()
    if not ret:
        print('No imgs grabbed!')
        break

    blank = np.zeros_like(img)
    blur_image = cv.GaussianBlur(img, (5, 5), 0)

    lower_hsv = np.array([78, 73, 0]) # lower threshold for H, S, and V respectively
    upper_hsv = np.array([180, 255, 255])

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, lower_hsv, upper_hsv)

    contours, h = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    def filterAspect(c):
        center, dimensions, angle = cv.minAreaRect(c)
        height, width = dimensions
        try:
            return width / height > 4 and cv.contourArea(c) > 5000        
        except:                                  
            return True
        
    filtered = list(filter(filterAspect, contours))
    if not filtered:
        continue
    cv.drawContours(img, filtered, -1, (0, 0, 255), 2)
    largest_contour = max(filtered, key=cv.contourArea)
    moments = cv.moments(largest_contour)

    # Find the centroid of the largest contour
    try:
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
    except:
        cv.imshow("f", img)
        cv.imshow("blank", blank)
        cv.waitKey(0)
        continue
    # Draw the centroid on the original image             
    newcenter =  (cx, cy)

    cv.circle(img, newcenter, 5, (0, 255, 0), -1)
    
    centersLog.append(newcenter)

    if centersLog:
        for i in range(len(centersLog) - 1):
            cv.line(img, centersLog[i], centersLog[i+1], (255, 255, 0), 2)
            cv.line(blank, centersLog[i], centersLog[i+1], (255, 255, 0), 2)

    cv.imshow("f", img)
    cv.imshow("blank", blank)
    cv.waitKey(0)

    
cv.destroyAllWindows()