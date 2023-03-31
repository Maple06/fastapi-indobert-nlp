import logging, logging.config
import pytz
from pytz import timezone
from datetime import datetime

tz = pytz.timezone('Asia/Jakarta')
logging.config.fileConfig('logging.ini')
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Jakarta')).timetuple()
logger = logging.getLogger('app')

# gunicorn_logger = logging.getLogger('gunicorn.error')
# logger.handlers = gunicorn_logger.handlers
# if __name__ != "main":
#     logger.setLevel(gunicorn_logger.level)
# else:
#     logger.setLevel(logging.DEBUG)

logger.info("Importing Libraries")
from fastapi import FastAPI, UploadFile, File, Form
from fastapi_utils.tasks import repeat_every
from time import strftime, gmtime
from apscheduler.schedulers.background import BackgroundScheduler
import face_recognition
import os
import shutil
import math
import requests
import cv2
import numpy as np
import dlib
logger.info("Libraries Imported!")

app = FastAPI()

scheduler = BackgroundScheduler()

status = {'faces': "No Face Detected", "confidence": "0", "match-status": False, "error-status": 1}
# Face detector from dlib
detector = dlib.get_frontal_face_detector()

recentPicTaken = ""

# Get face confidence of recognized image
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []

def update():
    logger.info("Updating datasets... (This may took a while)")
    # Grab images from web waktoo open API
    r = requests.get('https://web.waktoo.com/open-api/get-selfie?token=05e41dfb64d82ff61f50ec6691ab87fb', headers={'Accept': 'application/json'})

    # Get user IDs and all images from request
    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    response = response["data"][idPerusahaan-1]["user"]

    # Loop through users
    for i in response:
        try:
            count = 1
            # Loop through images per user
            for j in i["foto"]:
                url = j["foto_absen"]

                r = requests.get(url)

                filename = f'data/faces/{i["user_id"]}-pic{count}.png'

                # Save grabbed image to data/faces/
                with open(filename, 'wb') as f:
                    f.write(r.content)         
                try :
                    # Read recently grabbed image to cv2
                    img = cv2.imread(filename)

                    # Convert into grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Load the cascade
                    face_cascade = cv2.CascadeClassifier('ml-models/haarcascade_frontalface_default.xml')
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.4, 7)
                    
                    # Crop the faces
                    for (x, y, w, h) in faces:
                        faces = img[y:y + h, x:x + w]

                    # Upscale the image so face detection is more accurate
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    path = 'ml-models/FSRCNN_x4.pb'
                    sr.readModel(path)
                    sr.setModel("fsrcnn", 4)
                    upscaled = sr.upsample(faces)

                    # Save cropped image only if any face is detected 
                    # and the amount of images saved is less than 3
                    if count <= 3 :
                        cv2.imwrite(filename, upscaled)
                        count += 1
                    else :
                        try:
                            # If images saved is already at 3, remove the file even if there is a face
                            os.remove(filename)
                            break
                        except:
                            break
                except :
                    # Remove any images that doesn't contain a face
                    os.remove(filename)
                    break
        # Pass if user does not have any image in waktoo API
        except IndexError:
            pass

    logger.info("Datasets updated!")

todaysUserLen = 0

def encode_faces():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names, todaysUserLen

    # Get todays users count for real-time user count checking. (Update if there is any new user)
    r = requests.get('https://web.waktoo.com/open-api/get-selfie?token=05e41dfb64d82ff61f50ec6691ab87fb', headers={'Accept': 'application/json'})
    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    todaysUserLen = len(response["data"][idPerusahaan-1]["user"])

    # Update the dataset before encoding
    update()
    
    # Encoding faces (Re-training for face detection algorithm)
    logger.info("Encoding Faces... (This may take a while)")
    for image in os.listdir('data/faces'):
        face_image = face_recognition.load_image_file(f"data/faces/{image}")
        try:
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(image)
        except IndexError:
            pass
    
    logger.info("Encoding Done!")

# Encode face everytime server resets
encode_faces()

usersLen = 0

# Real-time checking on users count. If user is added, re-encode
@app.on_event("startup")
@repeat_every(seconds=10)
def getUsersLen():
    global usersLen, todaysUserLen

    r = requests.get('https://web.waktoo.com/open-api/get-selfie?token=05e41dfb64d82ff61f50ec6691ab87fb', headers={'Accept': 'application/json'})
    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    usersLen = len(response["data"][idPerusahaan-1]["user"])
    
    if usersLen != todaysUserLen:
        encode_faces()

# Re-encode dataset every single day on 1:00 a.m. local server time
logger.info("Running daily dataset update and encoding...")
scheduler.add_job(encode_faces, 'cron', day_of_week='mon-sun', hour=1, minute=00)
scheduler.start()
logger.info("Daily dataset updated and encoded!")

@app.post('/')
def api(user_id : str = Form(...), file: UploadFile = File(...)):
    logger.info("API Request Received")
    # If user id not provided return error
    if user_id == "":
        logger.warning("No user id provided")
        return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "No user id provided"}

    # Get time now for filename
    timeNow = strftime("%d-%b-%y.%H-%M-%S", gmtime())
    filename = f"data/images/api-{timeNow}.png"

    # In case any file currently processed have the same
    # filename, to anticipate error rename the file
    filenamesInImages = os.listdir("data/images")
    count = 0
    while filename.split("/")[-1] in filenamesInImages:
        filename = f"data/images/api-{timeNow}-{count}.png"
        count += 1

    # Save the image that is sent from the request and reject if filename is not valid
    with open(filename, "wb") as f:
        if file.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
            logger.warning("Filename not supported")
            return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "Filename not supported"}
        shutil.copyfileobj(file.file, f)

    # Get all IDs in dataset
    IDs_in_dataset = [''.join(list(set(i.split("-").pop()))) for i in os.listdir("data/faces/")]

    # Read image as cv2
    frame = cv2.imread(f"data/images/api-{timeNow}.png")

    try:
        if filename == None:
            logger.warning("Not a valid filename")
            return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "Not a valid filename"}
    except ValueError:
        pass

    global status

    status["faces"] = "No Face Detected"
    status["confidence"] = "0%"
    status["match-status"] = False
    status["error-status"] = 1

    # Upscale image
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'ml-models/FSRCNN_x4.pb'
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)
    
    # If image is too big, downscale. If image is too small, upscale. Else keep it as it is
    if frame.shape[0] >= 1000 or frame.shape[1] >= 1000:
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    elif frame.shape[0] <= 400 or frame.shape[1] <= 400 :
        small_frame = sr.upsample(frame)
        small_frame = cv2.resize(small_frame, (0, 0), fx=0.4, fy=0.4)
    else :
        small_frame = frame
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # If frame size not supported which cause 
    # the server to break, return error
    if rgb_small_frame.shape[0] > 600 :
        os.remove(filename)
        return {"faceDetected": status['faces'], "confidence": status["confidence"], "match-status": status["match-status"], "error-status": 0, "error-message": "File dimension not supported"}

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = '???'

        # Calculate the shortest distance to face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])

        status["faces"] = name
        status["confidence"] = confidence

        detectedFace = status["faces"].split("-")[0]
        if user_id == detectedFace:
            status["match-status"] = True

        face_names.append(f'{name} ({confidence})')

    if status["faces"] == "Unknown" and user_id not in IDs_in_dataset:
        os.remove(filename)
        logger.warning("User id not in dataset, but face is detected")
        return {"faceDetected": "Face Detected", "confidence": "N/A", "match-status": False, "error-status": 1, "error-message": "User id not in dataset, but face is detected"}

    # Display the results

    faceDetected = status["faces"]
    confidence = status["confidence"]

    # Delete image after process to save storage
    os.remove(filename)

    # Return all result if the process succeed
    logger.info("API return success")
    return {"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 1}