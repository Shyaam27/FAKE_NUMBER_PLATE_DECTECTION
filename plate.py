import numpy as np
import cv2
import imutils
import pytesseract
import mysql.connector
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Capture a frame from the webcam
ret, frame = cap.read()

# Resize the frame if necessary
frame = cv2.resize(frame, (0, 0), fx=500 / 400, fy=500 / 400)

# Process the captured image
gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_scaled = cv2.bilateralFilter(gray_scaled, 15, 20, 20)
edges = cv2.Canny(gray_scaled, 170, 200)

contours, heirarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = frame.copy()
cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]
Number_Plate_Contour = 0
for current_contour in contours:
    perimeter = cv2.arcLength(current_contour, True)
    approx = cv2.approxPolyDP(current_contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        Number_Plate_Contour = approx
        break
mask = np.zeros(gray_scaled.shape, np.uint8)
new_image1 = cv2.drawContours(mask, [Number_Plate_Contour], 0, 255, -1)
new_image1 = cv2.bitwise_and(frame, frame, mask=mask)

gray_scaled1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2GRAY)
ret, processed_img = cv2.threshold(np.array(gray_scaled1), 125, 255, cv2.THRESH_BINARY)

# Extract text from the number plate image
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Extract text from the number plate image and remove spaces
plate_text = pytesseract.image_to_string(processed_img).strip()
text = ''.join(plate_text.split())

# Display the detected number plate number
print("Detected Number Plate:", plate_text)

cv2.imshow("Processed Frame", frame)
cv2.waitKey(3000)  # Display for 3 seconds (adjust the delay as needed)

# Connect to the MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="numberplate"
)

# Create a cursor object to execute SQL queries
cursor = db.cursor()

# Perform a query to check if the scanned number is in the database
query = "SELECT * FROM number_plates WHERE plate_number = %s"
cursor.execute(query, (text,))
result = cursor.fetchone()

# Check if the number plate exists in the database
if result:
    print("Number plate is not fake.")
else:
    print("Number plate is fake.")

# Release the webcam, close the database connection, and destroy all windows
cap.release()
cursor.close()
db.close()
cv2.destroyAllWindows()
