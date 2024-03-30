import cv2
import numpy as np

# Video
cap = cv2.VideoCapture('video.mp4')

min_width_rect = 80
min_height_rect = 80

count_line_position = 550

algo=cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy
detect =[]
offset = 6 # Allowable error between pixel
counter = 0


while True:
    ret,frame=cap.read()
    # Converting to grayscale makes the task easier by removing color complexity, keeping things simple and 
    # uniform, and helping the program detect vehicles more reliably regardless of lighting.
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

    # Blurring reduces image noise and smooths out details, aiding in the detection of objects by creating a 
    # more uniform background.
    blur = cv2.GaussianBlur(grey,(5,5),5)

    # Applying Background subtraction on each frame by using cv2.createBackgroundSubtractorMOG2()
    img_sub = algo.apply(blur)

    # This line performs morphological dilation on a binary image to expand foreground regions, aiding in noise 
    # reduction and object detection.
    dilat = cv2.dilate(img_sub,np.ones((5,5)))

    # This line generates a kernel  of elliptical shape with a size of 5x5 pixels, which is commonly used for 
    # morphological operations such as dilation, erosion, opening, and closing in image processing tasks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # This line applies morphological closing to the binary image dilat using the specified elliptical kernel, 
    # resulting in smoother foreground objects with small holes filled in.
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # This line finds contours in the binary image dilatada using the specified retrieval mode (cv2.RETR_TREE) 
    # and contour approximation method (cv2.CHAIN_APPROX_SIMPLE), returning the contours and the contour hierarchy (counterShape and h, respectively).
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # This line draws a straight line on the frame image from point (25, count_line_position) to point 
    # (1200, count_line_position) with the specified color (255,127,0) and thickness 3.
    cv2.line(frame,(25,count_line_position),(1200, count_line_position),(255,127,0),3)

    # This portion of the code is responsible for detecting vehicles in the frame and counting them as they 
    # cross a specified line on the screen.
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter :
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+w),(0,0,255),2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4, (0,0,255),-1)


        for (x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame,(25,count_line_position),(1200, count_line_position),(0,127,255),3)
                detect.remove((x,y))

    cv2.putText(frame,"Vehicle Counter : "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    cv2.imshow("Detector",frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release 

