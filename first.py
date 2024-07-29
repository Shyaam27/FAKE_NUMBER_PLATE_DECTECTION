import numpy as np
import cv2
import imutils
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np

filename ='image1.jpg'
image = cv2.imread(filename)
image = cv2.resize(image, (0,0), fx=500/400, fy=500/400)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_scaled = cv2.bilateralFilter(gray_scaled, 15, 20, 20)
edges = cv2.Canny(gray_scaled, 170, 200)
#cv2.imshow("Edged", edges)
cv2.waitKey(0)
contours, heirarchy  = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = image.copy()
cv2.drawContours(img1, contours, -1, (0,255,0), 3)
#cv2.imshow("All of the contours", img1)
cv2.waitKey(0)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:50]
Number_Plate_Contour = 0
for current_contour in contours:        
    perimeter = cv2.arcLength(current_contour, True)
    approx = cv2.approxPolyDP(current_contour, 0.02 * perimeter, True)
    if len(approx) == 4:  
           Number_Plate_Contour = approx
           break
mask = np.zeros(gray_scaled.shape,np.uint8)
new_image1 = cv2.drawContours(mask,[Number_Plate_Contour],0,255,-1,)
new_image1 =cv2.bitwise_and(image,image,mask=mask)
#cv2.imshow("Number Plate",new_image1)
cv2.waitKey(0)
gray_scaled1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2GRAY)
ret,processed_img = cv2.threshold(np.array(gray_scaled1), 125, 255, cv2.THRESH_BINARY)
#cv2.imshow("Number Plate",processed_img)
cv2.waitKey(0)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

text = pytesseract.image_to_string(processed_img)
print("Number is :", text)
cv2.waitKey(0)
