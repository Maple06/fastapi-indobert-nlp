# Face Recognition Website using FastAPI, OpenCV, and face-recognition library.

### Usage
- Using docker <br>
`docker compose up --build`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 3345`
    - This runs the app on localhost port 3345

Send a post request to the main directory "/" (localhost:3345) that includes 2 body requests, "file" which is an image upload/image binary string and "user_id" which is the user ID to match (string).

This API updates then re-train datasets on 01:00 a.m. local time

### Outputs
- Json format: {"faceDetected": \<null/string/list\>, "confidence": \<null/string\>, "match-status": \<true/false\>, "error-status": \<0/1\>, "error-message": \<string\>} (If error-status is 0, error-message is not outputted)
1. "No user id provided"
    - The user_id request field is empty
2. "Filename not supported"
    - API only accepts jpg, png, jpeg, and heif. Set this in app/api/api_v1/services/facerecog_service.py line 36.
3. "No face detected"
    - API success. No face is detected.
4. "Found more than 1 face, but one face matched"
    - API success. Face detected is more than 1, but one face matched. Accept request is possible, but not recommended since it will confuse the next image training.
5. "Found more than 1 face, and none of the faces matched"
    - API success. Self explanatory, can be treated the same as the next (number 6) error output.
6. "Face detected but not in dataset"
    - API success. Face is detected but there is no matching data with dataset. Can also be treated as "No face detected" error. (number 3)
7. No error-message and faceDetected + confidence is not null
    - API success and outputs whether the face detected matched with the input user ID or not.a
#### TO SIMPLIFY, you can accept outputs with "match-status" = true and deny with "match-status" = false.

### This is a ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage, Waktoo Product.
