
import cv2
import os 

diretorio = 'imagens'

arquivos = os.listdir(diretorio)

detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for arq in arquivos:
    if arq.lower().endswith('.jpg') or arq.lower().endswith('.png') or arq.lower().endswith('.jpeg'):
      
        img = cv2.imread(diretorio+'/'+arq)
        img = cv2.resize(img , (700, 500), interpolation=cv2.INTER_AREA)


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
       
        faces = detector_face.detectMultiScale(img_gray, 
                                                scaleFactor= 1.1, 
                                                minNeighbors=7,
                                                minSize=(30,30)) 
        
        for x, y, w, h in faces:
            # cv2.rectangle(image, start_point, end_point, color, thickness) 
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 7 )
            
    
        altura = int(img.shape[0]/img.shape[1]+640)
        img_resized = cv2.resize(img, (640, altura), interpolation=cv2.INTER_CUBIC)
        
       
        cv2.imshow("imgs", img_resized)       
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
        







""" 
# camera = cv2.VideoCapture(0)

# detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# while True:
#     _, frame = camera.read()

#     if not _:
#         print(_)
#         print("Falha")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = detector.detectMultiScale(gray, scaleFactor=1.1,
#     minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
#     # print(len(faces))
#     # print(faces)
   
#     for x, y, w, h in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 3)

#     cv2.imshow("detectados", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows() 
# """