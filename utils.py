from ast import Return
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def predict(boxes):
    model = loadModel("emnist-93acc.model")
    f = open('json/matriz_1.json')
    data = json.load(f)

    labels = []
    target = []
    for i in range(26):
        labels.append([])
        target.append(chr(i + 65))
        for j in range(26):
            if j == i:
                labels[i].append(1)
            else:
                labels[i].append(0)

    c = 0
    v = []
    for box in boxes:
        a = cv2.GaussianBlur(box, (9, 9), 1)
        a = a / 255
        for j in range(a.shape[0]):
            for k in range(a.shape[1]):
                if a[j][k] <= 0.5:
                    a[j][k] = 255
                else:
                    a[j][k] = 0

        a = cv2.resize(a, (28, 28))
        a = img_to_array(a)
        a = np.expand_dims(a, 0)

        predictedIdxs = model.predict(a, verbose=0)[0]
        predictedIdxs = np.argmax(predictedIdxs, axis=0)
        v.append(target[predictedIdxs])
        c += 1

    
    # cv2.imshow("aaa", boxes[17])

    matriz = []
    for i in range(10):
        matriz.append([])
        for j in range(10):
            matriz[i].append(v[i*9 + j + i])

    print()
    print("PREDICTIONS")
    print("---------------------------------------------")
    resposta = []
    for i in range(len(matriz)):
        resposta.append([])
        for j in range(len(matriz[0])):
            resposta[i].append({"valor": matriz[i][j], "color": False})
            if matriz[i][j] == data["matriz"][i][j]:
                print(f'{bcolors.OKGREEN} {matriz[i][j]} {bcolors.ENDC} ', end = '')
            else:
                print(f'{bcolors.FAIL} {matriz[i][j]} {bcolors.ENDC} ', end = '')
        print()

    # for i in range(len(matriz)):
    #     print(matriz[i])
    # return

    print()
    for p in data["palavras"]:
        achou = False
        posicoes = []
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                if matriz[i][j] == p[0] and j + len(p) <= len(matriz[i]):
                    achou = True
                    for k in range(len(p)):
                        posicoes.append((i + 1, k + j + 1))
                        if matriz[i][k + j] != p[k]:
                            achou = False
                            posicoes = []
                            break
                if matriz[i][j] == p[0] and i + len(p) <= len(matriz) and achou == False:
                    achou = True
                    for k in range(len(p)):
                        posicoes.append((i + k + 1, j + 1))
                        if matriz[i + k][j] != p[k]:
                            achou = False
                            posicoes = []
                            break
                if achou == True:
                    break
            if achou == True:
                break
        if achou == True:
            # print(f'Found word {p} at: {posicoes}')
            for item in posicoes:
                lin = item[0] - 1
                col = item[1] - 1
                resposta[lin][col]["color"] = True
        # else:
        #     print(f"Can't find word {p}")

    print("ANSWERS")
    print("---------------------------------------------")
    for i in range(len(resposta)):
        for j in range(len(resposta[i])):
            if resposta[i][j]["color"]:
                print(f'{bcolors.OKBLUE} {matriz[i][j]}{bcolors.ENDC}  ', end = '')
            else:
                print(f' {matriz[i][j]}  ', end = '')
        print()

    print()




    # print("\n")
    # print(data["matriz"])

    # cv2.imshow("asfsgsevc", cv2.resize(boxes[32], (28, 28)))
    # print(tf.keras.metrics.recall())
    # print(predictedIdxs)
    # print(f"- [DEBUG] Prediction: {v[32]}")
    # predictedIdxs = np.argmax(predictedIdxs, axis=0)
    # print(f"prediction: {target[predictedIdxs]}")


def closestNumber(n, m):
    q = int(n / m)

    n1 = m * q

    if((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    if (abs(n - n1) < abs(n - n2)):
        return n1

    return n2


def loadModel(modelName):
    model = load_model(f'model/{modelName}')
    return model


def preProcessImg(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


# 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


# 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# Split the board in smaller pieces (boxes)
def splitBoardIntoBoxes(img):
    rows = np.vsplit(img, 10)
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 10)
        for box in cols:
            boxes.append(box)

    return boxes


# 4 - GET PREDECTIONS ON ALL IMAGES
# def getPredection(boxes, model):
#     result = []
#     for image in boxes:
#         # PREPARE IMAGE
#         img = np.asarray(image)
#         img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
#         img = cv2.resize(img, (28, 28))
#         img = img / 255
#         img = img.reshape(1, 28, 28, 1)
#         # GET PREDICTION
#         predictions = model.predict(img)
#         classIndex = model.predict_classes(img)
#         probabilityValue = np.amax(predictions)
#         # SAVE TO RESULT
#         if probabilityValue > 0.8:
#             result.append(classIndex[0])
#         else:
#             result.append(0)
#     return result


# 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1] / 10)
    secH = int(img.shape[0] / 10)
    for x in range(0, 10):
        for y in range(0, 10):
            if numbers[(y * 10) + x] != 0:
                cv2.putText(img, str(numbers[(y * 10) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


# 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1] / 10)
    secH = int(img.shape[0] / 10)
    for i in range(0, 10):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img


# 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver
