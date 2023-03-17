import cv2
import mediapipe as mp
import gesture as gs

video = cv2.VideoCapture(0)

handsAI = mp.solutions.hands.Hands(static_image_mode=False,
                                                    max_num_hands=1,
                                                    min_tracking_confidence=0.5,
                                                    min_detection_confidence=0.5)



mpDraw=mp.solutions.drawing_utils

gesture = gs.Gesture()
arm = gs.Arm()
tiks = 0








while True:
    ret, image = video.read()
    hand = handsAI.process(image)

    if hand.multi_hand_landmarks:
        for pointID, pos in enumerate(hand.multi_hand_landmarks[0].landmark):

            height, weight, _ = image.shape
            pointPosInScreenX, pointPosInScreenY = int(pos.x*weight), int(pos.y*height)
            cv2.circle(image, (pointPosInScreenX, pointPosInScreenY), 8, (0,255,0), cv2.FILLED)

            ###обновление всех данных для корректной работы gesture

            gesture.update_point_pos(hand.multi_hand_landmarks[0].landmark)
            arm.update_info(gesture.currentPointsPos)
            arm._update_info_hand_position()

            ### вывод проверяемого параметра

            print(arm.handDirection)


        mpDraw.draw_landmarks(image,hand.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    tiks += 1
    cv2.imshow('Test', image)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
