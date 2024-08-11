"""
This module finds the solution to a given sudoku problem.
Code credits: Tim Ruscica
More info: https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/


"""

def solve(bo):
    """
    Solves the Sudoku puzzle using backtracking algorithm.
    
    Args:
        bo (list of list of int): 2D list representing the Sudoku board.

    Returns:
         True if the board is solved, False otherwise.
    """
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False

def valid(bo, num, pos):
    """
    Checks if a number can be placed at a given position on the board.
    
    Args:
        bo (list of list of int): 2D list representing the Sudoku board.
        num (int): Number to place.
        pos (tuple of int): (row, col) position to place the number.

    Returns:
       True if the number can be placed, False otherwise.
    """
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False

    return True

def print_board(bo):
    """
    Prints the Sudoku board in a readable format.
    
    Args:
        bo (list of list of int): 2D list representing the Sudoku board.
    """
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

def find_empty(bo):
    """
    Finds an empty cell on the board.
    
    Args:
        bo (list of list of int): 2D list representing the Sudoku board.

    Returns:
        tuple of int or None: (row, col) position of the empty cell, or None if no empty cell is found.
    """
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None
