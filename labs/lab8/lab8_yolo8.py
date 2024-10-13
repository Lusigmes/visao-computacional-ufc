from ultralytics import YOLO
import cv2
import os

diretorio = 'imagens' 
arquivos = os.listdir(diretorio)

model = YOLO('yolov8n.pt')

for arquivo in arquivos:
    if arquivo.lower().endswith('.jpg') or arquivo.lower().endswith('.png') or arquivo.lower().endswith('.jpg'):
        
        img = cv2.imread(diretorio+'/'+arquivo)

        img = cv2.resize(img, (700,500), interpolation=cv2.INTER_AREA)

        results = model(img)
        print(results)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0 ,0), 2)

                class_id = int(box.cls[0])
                conf = box.conf[0]
                label = f"{model.names[class_id]} {conf:.2f}"

                cv2.putText(img, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow("imgs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



















""" 
# camera = cv2.VideoCapture(0)

# model = YOLO("yolov8n.pt")

# while True:
#     _, frame = camera.read()

#     results = model(frame)
#     for result in results:
#         # boxes = results.boxes
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
#             # detections = results.xyxy[0].cpu().numpy()  # Resultados [x1, y1, x2, y2, conf, class]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             class_id = int(box.cls[0])
#             conf = box.conf[0]
#             label = f"{model.names[class_id]} {conf:.2f}"

#             cv2.putText(frame, label, (x1, y1-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     cv2.imshow("detecado", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()
 """