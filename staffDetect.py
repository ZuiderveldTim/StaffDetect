# USAGE
# python staffDetect.py
# python staffDetect.py --video videos/example_01.mp4

# import the necessary packages
import imutils
import time
import cv2
import face_recognition
import glob

known_face_encodings = []
known_face_names = []
#grabs all the images from the img folder and uses them as reference
for filename in glob.glob('img/*.jpg'):
    faceimage = face_recognition.load_image_file(filename)
    known_face_encodings.append(face_recognition.face_encodings(faceimage)[0])
    #strips .jpg & img/ from the filename to get the name of the person
    filename = filename.replace('.jpg', '')
    filename = filename.replace('img/', '')
    known_face_names.append(filename)
face_locations = []
face_encodings = []
face_names = []
movement = "False"
counter = 0
#we are reading from webcam
camera = cv2.VideoCapture(0)
time.sleep(0.25)


# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()
    counter+=1
    movement = "False"
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=3)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:


        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        surface = w + h
        if surface > 500:
            roi = frame[y:y + h, x:x + w]
            small_frame = cv2.resize(roi, (0, 0), fx=0.20, fy=0.20)

            rgb_small_frame = small_frame[:, :, ::-1]

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            movement = "True"

            for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"


            # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    cv2.imshow(name, small_frame)
                    if name not in face_names:
                        face_names.append(name)
    # every 100 frames the background is reset
    if counter >= 100:
        firstFrame = None
        print("frame reset")
        counter = 0
        for name in known_face_names:
            cv2.destroyWindow(name)


    # draw the text and timestamp on the frame
    cv2.putText(frame, "Movement = " + movement, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



    # show the frame and record if the user presses a key
    cv2.imshow("Staff Detector", frame)
    key = cv2.waitKey(1) & 0xFF


    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    if key == ord("p"):
        print(face_names)


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
