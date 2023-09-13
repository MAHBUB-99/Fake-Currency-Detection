import numpy as np
import cv2
import pytesseract

# Ask the user for the image file path
image_path = "./../Resources/1000_real_90_1.jpg"
width = 800

try:
    # Open the image using Pillow
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    aspect_ratio = img.shape[1] / img.shape[0]
    height = int(width / aspect_ratio)
    resized_img = cv2.resize(img, (width, height))

    # Display the image
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Resized Image", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray= cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # text = pytesseract.image_to_string(gray)
    # print("Detected Text"+text)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 2)
    # cv2.imshow("Blurred Image", gray_blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(
        gray_blur, # Input image
        cv2.HOUGH_GRADIENT, # Method
        dp=1, # Inverse ratio of the accumulator resolution to the image resolution
        minDist=0.01, # Minimum distance between the centers of the detected circles
        param1=40, # Upper threshold for the internal Canny edge detector
        param2=25, # Threshold for center detection
        minRadius=1, # Minimum radius to be detected
        maxRadius=8 # Maximum radius to be detected
    )

    #if Circles are found, draw them on the gray image
    if circles is not None:
        circles = np.uint16(np.around(circles)) # Convert the (x, y) coordinates and radius of the circles to integers
        for circle in circles[0, :]:    # For each detected circle
            center = (circle[0], circle[1]) # Get the circle center
            # cv2.circle(gray_blur, center, 1, (0, 0, 100), 3)  # Draw the circle center
            radius = circle[2]  # Get the circle radius
            cv2.circle(resized_img, center, radius, (0, 0, 255), 2) # Draw the circle outline
        
        cv2.imshow("Circles", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles were found")

except Exception as e:
    print(f"An error occurred: {e}")



