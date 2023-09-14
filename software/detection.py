import cv2

inputVideo = cv2.VideoCapture(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

while True:
    ret, image = inputVideo.read()
    if not ret:
        break

    imageCopy = image.copy()
    corners, ids, rejected = detector.detectMarkers(image)

    # if at least one marker detected
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(imageCopy, corners, ids)
        #cv2.aruco.drawDetectedDiamonds(imageCopy,corners,ids)
        #cv2.aruco.drawDetectedCornersCharuco(imageCopy,corners,ids)

    cv2.imshow("out", imageCopy)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

inputVideo.release()
cv2.destroyAllWindows()
