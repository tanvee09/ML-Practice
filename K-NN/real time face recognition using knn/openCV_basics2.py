import cv2

img = cv2.imread('../basics/logo.png')
gray = cv2.imread('../basics/logo.png', cv2.IMREAD_GRAYSCALE)


# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# matplotlib will show BGR image
# But imshow() of cv2 will interpret it as RGB only
cv2.imshow('Logo Image', img)
cv2.imshow('Gray Logo Image', gray)
cv2.waitKey(0) # Wait for infinite time
# cv2.waitKey(2500) # means destroy window after 2500 millisecs
cv2.destroyAllWindows()