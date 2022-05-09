import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms
import pandas as pd


# color boundaries
LOWER_HSV = {
    'red': np.array([145, 100, 20], np.uint8),
    "orange": np.array([10, 100, 100], np.uint8),
    'yellow': np.array([17, 100, 20], np.uint8),
    'green': np.array([60, 100, 20], np.uint8),
    'blue': np.array([90, 100, 20], np.uint8),
    "navy": np.array([110, 100, 20], np.uint8),
    "purple": np.array([125, 100, 20], np.uint8),
    "pink": np.array([0, 70, 20], np.uint8),
    "black": np.array([0, 0, 0], np.uint8), 
    "white": np.array([0, 0, 80], np.uint8),
    "grey": np.array([0, 0, 150], np.uint8), 
    "brown": np.array([10, 100, 20], np.uint8), 
}

UPPER_HSV = {
    'red': np.array([180, 255, 255], np.uint8),
    "orange": np.array([20, 255, 255], np.uint8),
    'yellow': np.array([35, 255, 255], np.uint8),
    'green': np.array([90, 255, 255], np.uint8),
    'blue': np.array([110, 255, 255], np.uint8),
    "navy": np.array([125, 255, 255], np.uint8),
    "purple": np.array([135, 255, 255], np.uint8),
    "pink": np.array([6, 255, 250], np.uint8),
    "black": np.array([180, 60, 80], np.uint8), 
    "white": np.array([120, 40, 177], np.uint8),
    "grey": np.array([40, 30, 170], np.uint8), 
    "brown": np.array([20, 255, 200], np.uint8) 
}

def calculate_area(image, image_area, color):
    result = 0
    kernal = np.ones((5, 5), 'uint8')

    # Convert BGR color space to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define masks for each color
    mask = cv2.inRange(hsv_image,
        LOWER_HSV[color], UPPER_HSV[color])

    # Create contour
    mask = cv2.dilate(mask, kernal)
    cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Track color
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > image_area * 0.001):
            _, _, w, h = cv2.boundingRect(contour)
            result += w*h

    return result

def detect_color(image_path, leftup=None, rightdown=None):
    # Get image
    image = image_path
#     image = cv2.imread(image_path)
    if leftup != None and rightdown != None:
        image = image[leftup[1]: rightdown[1], leftup[0]: rightdown[0]]

    height, width, _ = image.shape
    image_area = height * width

    # Calcuate color areas
    white_area = calculate_area(image, image_area, 'white')
    red_area = calculate_area(image, image_area, 'red')
    green_area = calculate_area(image, image_area, 'green')
    blue_area = calculate_area(image, image_area, 'blue')
    yellow_area = calculate_area(image, image_area, 'yellow')
    orange_area = calculate_area(image, image_area, 'orange')
    black_area = calculate_area(image, image_area, 'black')
    grey_area = calculate_area(image, image_area, 'grey')
    brown_area = calculate_area(image, image_area, 'brown')
    navy_area = calculate_area(image, image_area, 'navy')
    purple_area = calculate_area(image, image_area, 'purple')
    pink_area = calculate_area(image, image_area, 'pink')

    areas = {'white': white_area, 'grey': grey_area, 'black': black_area, 'brown': brown_area,
             'blue': blue_area, 'navy': navy_area, 'purple': purple_area, 'green': green_area, 'red': red_area,
             'orange': orange_area, 'yellow': yellow_area, 'pink': pink_area}
    
    return max(areas, key=lambda x : areas[x])

def detect_feature(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height, img_width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416),  swapRB=True, crop=False) #mean=(0,0,0)

    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)


    # Showing informations on the screen
    class_ids = []
    confidences = [] 
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.01: #0.01 

                # Object detected
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)

                # Rectangle coordinates
                xx = int(center_x - width / 2)
                yy = int(center_y - height / 2)

                boxes.append([xx, yy, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    draw_img = img.copy()
    count=0
    max_temp = 0

    ## NMS 처리하기
    conf_threshold = 0.1
    nms_threshold = 0.4
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    feature_list = []
    

    if len(idxs) > 0:
        for i in idxs.flatten():
            each_feature = []
            box = boxes[i]
            if box[1] < 0: box[1] = 0
            if box[0] < 0: box[0] = 0
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            if width*height > max_temp:
                max_temp = width*height
                cv2.rectangle(draw_img, (int(left), int(top)), (int(left+width), int(top+height)), color=(0,255,0), thickness=2)

                crop_img = img[top:top + height, left:left + width]
                color = detect_color(crop_img)
                clothes_type = YOLO_LABELS[class_ids[i]]
                
                cv2.imwrite("crop_img.jpg", crop_img)
                crop_img = tfms(Image.open("crop_img.jpg")).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(crop_img)

                for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    
                style = class_names[idx]
    
                each_feature.append(color)
                each_feature.append(style)
                each_feature.append(clothes_type)
                each_feature.append(color + " " + style)
                each_feature.append(color + " " + clothes_type)
                each_feature.append(style + " " + clothes_type)
                each_feature.append(color + " " + style + " " + clothes_type)
                

                count += 1
                feature_list.append(each_feature)
            else: 
                continue
    
    return feature_list





# YOLO 설정 파일 Path
labelsPath = os.getcwd()+"\\pretrained_yolo\\df2.names" # Hand 라벨
weightsPath = os.getcwd()+"\\pretrained_yolo\\yolov3-df2_15000.weights" # 가중치
configPath = os.getcwd()+"\\pretrained_yolo\\yolov3-df2.cfg" # 모델 구성

# YOLO 라벨(hand) 호출
YOLO_LABELS = open(labelsPath).read().strip().split("\n")

# YOLO 모델 호출
yolo_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO 출력층 설정
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(YOLO_LABELS), 3))
frame_id = 0
count=0


model_name = 'efficientnet-b0'  # b5

image_size = EfficientNet.get_image_size(model_name)
men_model = EfficientNet.from_pretrained(model_name, num_classes=4)
women_model = EfficientNet.from_pretrained(model_name, num_classes=4)
MEN_PATH = "./pretrained_efficientnet/men_style_discriminator.pt"
WOMEN_PATH = "./pretrained_efficientnet/women_style_discriminator.pt"
device = torch.device('cpu') 

men_model.load_state_dict(torch.load(MEN_PATH, map_location=device))
women_model.load_state_dict(torch.load(WOMEN_PATH, map_location=device))

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

class_names = ["Casual", "Hip", "Office", "Sports"]

women_list = []
men_list = []

for i in range(1, 1201):
    try:
        men_list.extend(33(f"./crawled_images/shein_men_crawled\\best{i}.jpg", men_model)) 
        men_list.extend(detect_feature(f"./crawled_images/wconcept_men_crawled\\best{i}.jpg", men_model))
        women_list.extend(detect_feature(f"./crawled_images/shein_women_crawled\\best{i}.jpg", women_model))
        women_list.extend(detect_feature(f"./crawled_images/wconcept_women_crawled\\best{i}.jpg", women_model))
    except:
        continue

men_df = pd.DataFrame(men_list)
men_df.columns = ["color", "style", "type", "color+style", "color+type", "style+type", "color+style+type"]
# print(men_df.head)
men_df.to_excel('men.xlsx', index=False) 

women_df = pd.DataFrame(women_list)
women_df.columns = ["color", "style", "type", "color+style", "color+type", "style+type", "color+style+type"]
# print(women_df.head)
women_df.to_excel('women.xlsx', index=False)