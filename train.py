import cv2
import os
import numpy as np

def collect_data():
    images=[]
    labels=[]
    label_dic={}
    people=[person for person  in os.listdir("F:/AI-Projects/Face_Recognisation_Haarcascades/data/people/")]
    for i , person in enumerate(people):
        label_dic[i]=person
        for  image in os.listdir("F:/AI-Projects/Face_Recognisation_Haarcascades/data/people/"+person):
            images.append(cv2.imread("F:/AI-Projects/Face_Recognisation_Haarcascades/data/people/"+person+'/'+image,0))
            labels.append(i)
    return (images,np.array(labels),label_dic)




