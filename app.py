from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from utils import *
import sudoku_solver

import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Make sure the 'static/uploads' directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize the CNN model for digit prediction
model = intializePredectionModel()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('solve_sudoku', filename=file.filename))
    return render_template('index.html')

@app.route('/solve/<filename>')
def solve_sudoku(filename):
    pathImage = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    heightImg = 450
    widthImg = 450

    # 1. PREPARE THE IMAGE
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # Resize image to make it a square
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Create a blank image for debugging if required
    imgThreshold = preProcess(img)  # Preprocess the image

    # 2. FIND ALL CONTOURS
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # Draw all detected contours

    # 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        numbers = getPredection(boxes, model)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)

        # 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers, 9)
        try:
            sudoku_solver.solve(board)  # Solve the Sudoku puzzle
        except:
            pass

        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers = flatList * posArray
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

        # 6. OVERLAY SOLUTION
        pts2 = np.float32(biggest)
        pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgInvWarpColored = img.copy()
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)

        # Save the result image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'solved_' + filename)
        cv2.imwrite(result_path, inv_perspective)

        return render_template('result.html', original_image=filename, solved_image='solved_' + filename)

    else:
        return "No Sudoku Found"

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
