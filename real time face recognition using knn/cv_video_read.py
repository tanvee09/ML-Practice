# Read a video stream from camera (frame by frame)

import cv2

# Capture the device from which you want to read images
cap = cv2.VideoCapture(0) # 0 means default webcam


while True :
    
    ret, frame = cap.read() # ret is true if image is captured properly
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Video Frame1", gray_frame)
    # cv2.imshow("Video Frame2", frame)
    
    #Wait for user input - enter q to stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF 
    #cv2.waitKey gives 32bit int
    #0xFF is 8bit ones
    #& gives last bits -> convert 32bits to 8bits since ascii values : 0-255(8bit)
    if key_pressed == ord('q') :
        break

cap.release() # release device
cv2.destroyAllWindows()