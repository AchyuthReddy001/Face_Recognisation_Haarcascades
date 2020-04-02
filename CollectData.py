import cv2

def cut_faces(frame,face_coords):
    faces=[]

    for (x,y,w,h) in face_coords:
        w_rm=int(0.2 * w/2)
        faces.append(frame[y:y+h,x+w_rm:x+w-w_rm])
    return faces

def normalize_intensity(images):
    image_norm=[]
    for image in images:
        is_color= len(image.shape) == 3

        if is_color:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_norm.append(cv2.equalizeHist(image))
    return image_norm

def resize(images,size=(50,50)):
    images_norm=[]
    for image in images:
        if image.shape < size:
            image_norm=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        else:
            image_norm=cv2.resize(image,size,interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm


def normalize_faces(frame,face_coords):
    faces=cut_faces(frame,face_coords)
    faces=normalize_intensity(faces)
    faces=resize(faces)

    return faces

def draw_rectangles(image,coords):
    for (x,y,w,h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image,(x+w_rm,y),(x+w-w_rm,y+h),(150,150,0),8)
        


