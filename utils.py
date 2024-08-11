import cv2
import numpy as np
from tensorflow.keras.models import load_model

#### READ THE MODEL WEIGHTS
def intializePredectionModel():
    """
    Initialize the prediction model by loading the pre-trained weights.
    
    Returns:
        model: Loaded CNN model.
    """
    model = load_model('resources/model.h5')
    return model

#### 1 - Preprocessing Image
def preProcess(img):
    """
    Preprocess the input image by converting to grayscale, applying Gaussian blur, and adaptive threshold.
    
    Args:
        img (numpy array): Input image.
    
    Returns:
        imgThreshold (numpy array): Preprocessed image.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold

#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    """
    Reorder points to prepare for warp perspective transformation.
    
    Args:
        myPoints (numpy array): Array of points.
    
    Returns:
        myPointsNew (numpy array): Reordered points.
    """
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#### 3 - FINDING THE BIGGEST CONTOUR ASSUMING THAT IS THE SUDOKU PUZZLE
def biggestContour(contours):
    """
    Find the biggest contour which is assumed to be the Sudoku puzzle.
    
    Args:
        contours (list of numpy arrays): List of contours.
    
    Returns:
        tuple: Biggest contour and its area.
    """
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

#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFERENT IMAGES
def splitBoxes(img):
    """
    Split the image into 81 different boxes (each representing a cell in the Sudoku grid).
    
    Args:
        img (numpy array): Input image.
    
    Returns:
        boxes (list of numpy arrays): List of 81 image boxes.
    """
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

#### 4 - GET PREDICTIONS ON ALL IMAGES
def getPredection(boxes, model):
    """
    Get predictions for each box using the CNN model.
    
    Args:
        boxes (list of numpy arrays): List of image boxes.
        model (keras model): Loaded CNN model.
    
    Returns:
        result (list of int): List of predicted numbers.
    """
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

#### 6 - TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img, numbers, color=(0, 255, 0)):
    """
    Display the solved numbers on the image.
    
    Args:
        img (numpy array): Input image.
        numbers (list of int): List of numbers to display.
        color (tuple of int): Color of the text.
    
    Returns:
        img (numpy array): Image with numbers displayed.
    """
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

#### 6 - DRAW GRID TO SEE THE WARP PERSPECTIVE EFFICIENCY (OPTIONAL)
def drawGrid(img):
    """
    Draw a grid on the image to visualize the warp perspective efficiency.
    
    Args:
        img (numpy array): Input image.
    
    Returns:
        img (numpy array): Image with grid drawn.
    """
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for i in range(0, 9):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img

#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale):
    """
    Stack all images in one window for display.
    
    Args:
        imgArray (list of list of numpy arrays): 2D list of images to stack.
        scale (float): Scaling factor for the images.
    
    Returns:
        ver (numpy array): Stacked images.
    """
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
