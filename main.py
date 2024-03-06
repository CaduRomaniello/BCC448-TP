# import sudukoSolver
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF debugg log
from utils import *


pathImage = "img/cacaPalavras1.png"

# 1. Load Image, and pre-process it
print("- [INFO] Loading Image ...")
img = cv2.imread(pathImage)
heightImg, widthImg, _ = img.shape
heightImg = closestNumber(heightImg, 10)
widthImg = closestNumber(widthImg, 10)

img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcessImg(img)

# 2. FIND ALL COUNTOURS
print("- [INFO] Finding countours ...")
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)  # DRAW ALL DETECTED CONTOURS

# # 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
# print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    print("- [INFO] Feeding the image to a neural network ...")
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoardIntoBoxes(imgWarpColored)

    # Return the letter matrix
    letterMatrix = predict(boxes)
    # nimg = displayPredictedLetters(nimg, letterMatrix, color=(255, 0, 255))  # originally displayNumbers

    # Ideia, mostrar em nimg os caracteres que ele reconheceu, e trocar de cor os que formam as palavras, caso formem alguma

    # #### 5. FIND SOLUTION OF THE BOARD
    # board = np.array_split(numbers,9)
    # print(board)
    # try:
    #     sudukoSolver.solve(board)
    # except:
    #     pass
    # print(board)
    # flatList = []
    # for sublist in board:
    #     for item in sublist:
    #         flatList.append(item)
    # solvedNumbers =flatList*posArray
    # imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # # #### 6. OVERLAY SOLUTION
    # pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    # pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    # imgInvWarpColored = img.copy()
    # imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    # inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    # imgDetectedDigits = drawGrid(imgDetectedDigits)
    # imgSolvedDigits = drawGrid(imgSolvedDigits)

    # imageArray = [img, nimg]
    # cv2.imshow('Original Image', cv2.resize(imageArray[0], (315, 315)))
    # cv2.imshow('Solved Image', cv2.resize(imageArray[1], (315, 315)))

else:
    print("- [WARNING] COULD NOT FIND ANY BOARD IN THE IMAGE.")

cv2.waitKey(0)
