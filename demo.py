import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_mask = cv2.imread('images/potato.jpg')

webcam_capture = cv2.VideoCapture(0)
def capture_webcam():
    ret, frame = webcam_capture.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, grayscale_frame

def draw_boxes(frame, grayscale_frame): 
    faces = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grayscale_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

def draw_potato(frame, grayscale_frame):
    faces = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)
    for (x,y,w,h) in faces:
        if h > 0 and w > 0:
            # Extract the region of interest from the image
            frame_roi = frame[y:y+h, x:x+w]
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)

            # Convert color image to grayscale and threshold it
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 250, 255, cv2.THRESH_BINARY_INV)

            # Create an inverse mask
            mask_inv = cv2.bitwise_not(mask)

            # Use the mask to extract the face mask region of interest

            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

            # Use the inverse mask to get the remaining part of the image
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)

            # add the two images to get the final output
            frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)


def handle_modes(key_pressed, frame, grayscale_frame):
    if key_pressed == ord('b'):
        draw_boxes(frame, grayscale_frame)
    elif key_pressed == ord('p'):
        draw_potato(frame, grayscale_frame)

key_pressed = ord('b')
while (True):
    print(key_pressed)
    frame, grayscale_frame = capture_webcam()
    
    handle_modes(key_pressed, frame, grayscale_frame)
    cv2.imshow('img',frame)
    temp_key_pressed = cv2.waitKey(10)

    #update if key is pressed    
    if temp_key_pressed != -1:
        key_pressed = temp_key_pressed

    #stop program if x is pressed
    if  key_pressed == ord('x'):
        break

cv2.destroyAllWindows()
webcam_capture.release()