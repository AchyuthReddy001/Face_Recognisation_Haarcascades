import os
from Fdetect import VideoCamera,FaceDetect
import cv2
from awscli.compat import raw_input
import CollectData as cd
webcam=VideoCamera(0)
detector=FaceDetect("F:\\AI-Projects\\Face_Recognisation_Haarcascades\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt.xml")
floder="F:/AI-Projects/Face_Recognisation_Haarcascades/data/people/"+raw_input('Person: ').lower()
cv2.namedWindow("LoadData",cv2.WINDOW_AUTOSIZE)
if not os.path.exists(floder):
    os.mkdir(floder)
    counter=0
    timer=0
    while counter  < 10:
        frame=webcam.get_frame()
        faces_cords=detector.detect(frame)
        if len(faces_cords) and timer % 700 ==50 :
            faces=cd.normalize_faces(frame, faces_cords)
            cv2.imwrite(floder+'/'+str(counter)+'.jpg',faces[0])
            counter+=1
        cd.draw_rectangles(frame,faces_cords)
        cv2.imshow("TestData",frame)
        cv2.waitKey(50)
        timer+=50
    cv2.destroyAllWindows()
else:
    print("User alredy exist")
