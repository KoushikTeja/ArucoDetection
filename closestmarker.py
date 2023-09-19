import cv2 as cv
from cv2 import aruco
import numpy as np
import time

calib_data_path = r"MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 8  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()


def markclosest():
    cap = cv.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )
        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            total_markers = range(0, marker_IDs.size)
            dictmarkers = {}
            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                if marker_corners:
                    # Print the detected ArUco marker IDs as a list
                    detected_markers = [ids[0] for ids in marker_IDs]
                    print("Detected ArUco markers:", detected_markers)

                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                distance = np.sqrt(
                    tVec[i][0][2] ** 2 +
                    tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                dictmarkers[ids[0]] = distance
                print(dictmarkers, ids[0])
                temp = min(dictmarkers.values())
                res = [key for key in dictmarkers if dictmarkers[key] == temp]
                print(ids[0],res)
                if ids[0] == min(res):
                    cv.polylines(
                        frame, [corners.astype(
                            np.int32)], True, (255, 0, 255), 4, cv.LINE_AA
                    )
                    cv.putText(
                        frame,
                        f"id: {ids[0]} Dist: {round(distance, 2)}",
                        top_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (255, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                else:
                    cv.polylines(
                        frame, [corners.astype(
                            np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    cv.putText(
                        frame,
                        f"id: {ids[0]} Dist: {round(distance, 2)}",
                        top_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        frame,
                        f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                        bottom_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                # Draw the pose of the marker
                point = cv.drawFrameAxes(
                    frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        time.sleep(1)
    cap.release()
    cv.destroyAllWindows()

markclosest()
