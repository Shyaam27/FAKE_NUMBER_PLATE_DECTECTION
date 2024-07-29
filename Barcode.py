import numpy as np
import cv2
import imutils
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np

import barcode
from barcode.writer import ImageWriter
import pandas as pd
from pyzbar import pyzbar

filename ='img.jpg'
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





code128 = barcode.get_barcode_class('code128')
path_to_csv = r"Number.csv"
path_of_folder = r"Barcode" + chr(92)
df = pd.read_csv(path_to_csv)

def generate_barcodes():
    for index, row in df.iterrows():
        bar_code = row['Number']
        codes128 = code128(bar_code, writer=ImageWriter())
        codes128.save(path_of_folder + bar_code)

def read_barcodes(frame):
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{barcode_data} ({barcode_type})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if barcode_data in df['Number'].values:
            print(f"Barcode {barcode_data} is not fake")
        
        
        else:
            print(f"Barcode {barcode_data} is fake")
            
    return frame


def main():
    generate_barcodes()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame = read_barcodes(frame)
        cv2.imshow('Barcode Reader', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

