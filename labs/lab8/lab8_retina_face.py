from retinaface import RetinaFace
import cv2
import os

diretorio = "imagens"

arquivos = os.listdir(diretorio)

model = RetinaFace.build_model()

for arquivo in arquivos:
    if arquivo.lower().endswith('.jpg') or arquivo.lower().endswith('.png') or arquivo.lower().endswith('.jpeg'):

        img = cv2.imread(diretorio+'/'+arquivo)
        img = cv2.resize(img , (700, 500), interpolation=cv2.INTER_AREA)

        faces = RetinaFace.detect_faces(img,model=model)

        if isinstance(faces, dict):
            for face_id, face_info in faces.items():
                #extrair coordenadas da bbox
                face = face_info["facial_area"]
                x1,y1, x2,y2 = face
                #desenhar bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                #detectar kp do rosto
                face_kps = face_info['landmarks']
                for key, point in face_kps.items():
                    cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0 ,255), -1)

        cv2.imshow("imgs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()













""" 
# camera = cv2.VideoCapture(0)

# # rf_detector = RetinaFace(quality="normal")

# while True:
#     _, frame = camera.read()

# # Detecta rostos no frame
#     faces = RetinaFace.detect_faces(frame)

#     # Verifica se algum rosto foi detectado
#     if isinstance(faces, dict):
#         for face_id, face_info in faces.items():
#             # Extrai as coordenadas da caixa delimitadora do rosto
#             facial_area = face_info["facial_area"]
#             x1, y1, x2, y2 = facial_area
            
#             # Desenha um retângulo ao redor do rosto detectado
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Detecta pontos chave (olhos, nariz, boca)
#             landmarks = face_info["landmarks"]
#             for key, point in landmarks.items():
#                 cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)


#     cv2.imshow("frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()



# # Resumo do RetinaFace: O RetinaFace é um detector de faces de 
# # alta precisão e em tempo real, baseado em redes neurais profundas.
# # Ele foi projetado para detectar não apenas a localização das 
# # faces, mas também pontos-chave faciais como olhos, nariz, e
# # lábios, facilitando tarefas como alinhamento de rosto. 
# # Ele funciona em diferentes resoluções de imagem e pode 
# # ser configurado para diferentes prioridades, 
# # como velocidade ou precisão.

# # Formato de cor aceito: RetinaFace aceita imagens no formato RGB.
# # Se sua imagem estiver em formato BGR (como é o caso das imagens
# # lidas pelo OpenCV), é necessário converter para RGB antes de 
# # passar ao detector, usando cv2.cvtColor(img, cv2.COLOR_BGR2RGB).

# # Outros formatos como HSV ou grayscale não são diretamente 
# # suportados, pois ele depende da informação de cor RGB para
# # a detecção. 
# """