from ultralytics import YOLO
import cv2
import pytesseract
import re
from deep_translator import GoogleTranslator
import textwrap
from PIL import Image, ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO("train/runs/train1/weights/best.pt") #pega o meu modelo treinado
#model = YOLO("scr_manga/weights_AymanKUMA/best.pt") #pega o modelo que achei na internet
image = cv2.imread("scr_manga/inputs/2.jpeg") #carrega a imagem
results = model(image) #passa a imagem pro modelo e recebe o resultado
cv2.imwrite("scr_manga/outputs/resultado.jpg", results[0].plot()) #coloca a imagem resultante na pasta outputs


image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #converte do opencv para pil
draw = ImageDraw.Draw(image_pil) #objeto de desenho
fonte = ImageFont.truetype("arial.ttf", size=20)

#balloon = len(results[0].boxes) #numero de baloes de fala

for i,box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordenadas

    cropped = image[y1:y2, x1:x2] #recorta a imagem
    config = r'--oem 3 --psm 6' #vi em um video e melhorou o resultado kk
    text = pytesseract.image_to_string(cropped,config=config) #pega o texto da imagem
    text = re.sub(r'\s+', ' ', text).strip() #remove quebra de linha
    text = GoogleTranslator(source='en', target='pt').translate(text) #traduz balão
    text = text.capitalize() #formata o texto
    #print(f"Balão {i+1}: {text}")

    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) #passa o "branco"

    caract = max(((x2 - x1) // 10), 1) #maximo num de caracteres, 10 pixels por caracter
    linhas = textwrap.wrap(text, width=caract) #quebra o texto em linhas

    for linha in linhas: #para cada linha escreve na imagem um com uma margem vertical e horizontal
        draw.text((x1 + 5, y1 + 5), linha, font=fonte, fill=(0, 0, 0))
        y1 += 25

image_pil.save("scr_manga/outputs/2.jpeg") #salva a imagem com as alterações

