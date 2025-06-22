from ultralytics import YOLO
import cv2
import pytesseract
import re
from deep_translator import GoogleTranslator
import textwrap
#from PIL import Image, ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO("train/runs/detect/train8/weights/best.pt") #pega o meu modelo treinado
#model = YOLO("scr_manga/weights/best.pt") #pega o modelo que achei na internet
image = cv2.imread("scr_manga/inputs/img.jpeg") #carrega a imagem
results = model(image) #passa a imagem pro modelo e recebe o resultado
#cv2.imwrite("scr_manga/outputs/1.jpeg", results[0].plot()) #coloca a imagem resultante na pasta outputs

balloon = len(results[0].boxes) #numero de baloes de fala

for i,box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordenadas

    cropped = image[y1:y2, x1:x2] #recorta a imagem
    config = r'--oem 3 --psm 6' #Vi em um video e melhorou o resultado kk
    text = pytesseract.image_to_string(cropped,config=config) #pega o texto da imagem
    text = re.sub(r'\s+', ' ', text).strip() #remove quebra de linha
    text = GoogleTranslator(source='en', target='pt').translate(text)#traduz balão

    
    
    print(f"Balão {i+1}: {text}")
    cv2.imwrite(f"scr_manga/outputs/recorte_{i}.jpg", cropped) #salva o balão

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1) #Passa um "branco" no balão



    largura_balao = x2 - x1
    font_scale = 0.8
    max_caracteres_por_linha = max((largura_balao // 10), 1)
    linhas = textwrap.wrap(text, width=max_caracteres_por_linha)

    # Desenha o texto no balão
    y_texto = y1 + 20
    for linha in linhas:
        cv2.putText(
            image, linha, (x1 + 5, y_texto),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        y_texto += int(20 * font_scale)  # próxima linha




cv2.imwrite("scr_manga/outputs/resultado.jpg", image) #salva as imagens com as alterações

