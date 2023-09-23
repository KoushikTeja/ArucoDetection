import cv2

inputVideo = cv2.VideoCapture(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
detected_ids_list = []  # Initialize an empty list to store detected IDs for each frame

while True:
    ret, image = inputVideo.read()
    if not ret:
        break

    imageCopy = image.copy()
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None and len(ids) > 0:
        detected_frame_ids = [id[0] for id in ids]  # Extract IDs from the list
        detected_ids_list.append(detected_frame_ids)  # Store IDs for this frame

        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)

        # Print detected IDs for this frame
        print("Detected IDs:", detected_frame_ids)

    cv2.imshow("out", imageCopy)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

inputVideo.release()
cv2.destroyAllWindows()

print("Detected IDs for each frame:")
for i, frame_ids in enumerate(detected_ids_list):
    print(f"Frame {i + 1}: {frame_ids}")
