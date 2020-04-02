import cv2
from Fdetect import FaceDetect,VideoCamera
import train as t
#import createDataset as cds
import CollectData as cd

webcam=VideoCamera()
detector=FaceDetect("F:\\AI-Projects\\Face_Recognisation_Haarcascades\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt.xml")

#cv2.namedWindow("LiveFootage",cv2.WINDOW_AUTOSIZE)
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))
while  True:
    frame=webcam.get_frame()
    face_cords=detector.detect(frame,True)
    if len(face_cords) :
        faces=cd.normalize_faces(frame,face_cords)
        #print(faces)
        for i,face  in enumerate(faces):
            collector=cv2.face.StandardCollector_create()
            images, labels, label_dic = t.collect_data()
            rec_rig = cv2.face.LBPHFaceRecognizer_create()
            rec_rig.train(images, labels)
            rec_rig.predict_collect(face,collector)
            conf=collector.getMinDist()
            pred=collector.getMinLabel()
            thersold=160
            print(conf)
            if conf < thersold :
                cv2.putText(frame, label_dic[pred].capitalize(),
                            (face_cords[i][0], face_cords[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2)
            else:
                cv2.putText(frame, "UnKnown",
                            (face_cords[i][0], face_cords[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2)
        cd.draw_rectangles(frame,face_cords)
    out.write(frame)
    cv2.imshow("LiveVideo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break





