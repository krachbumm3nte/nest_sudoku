
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


# RGB values for the colors used in the output graphic
green =     (0, 105, 0)
red =       (150, 0, 0)
dark_grey = (200, 200, 200)
black =     (0,0,0)
white =     (255,255,255)

font = font_manager.FontProperties(family='sans')
#TODO: failsafe
font_loc = font_manager.findfont(font)

font_digits = ImageFont.truetype(font_loc, 18, encoding="unic")
font_text = ImageFont.truetype(font_loc, 14, encoding="unic")


def get_puzzle(puzzle_index):
    """returns one of 8 sudoku configuration to be solved.

    Args:
        puzzle_index (int): index between 0 and 7 indicating the puzzle number

    Returns:
        np.array: array of shape (9,9) representing the puzzle configuration.
        Array is zero wherever no input is given, and contains the corresponding
        digit otherwise.
    """
    init_config = None

    if not 0 <= puzzle_index < 8:
        raise ValueError(
            "Cannot return puzzle - index must be between 0 and 7!")

    if puzzle_index == 0:
        # Dream problem: make the network come up with a valid sudoku without any restrictions
        init_config = np.zeros((9, 9), dtype=np.uint8)
    elif puzzle_index == 1:
        # Diabolical problem:
        init_config = [[0, 0, 1,  0, 0, 8,  0, 7, 3],
                       [0, 0, 5,  6, 0, 0,  0, 0, 1],
                       [7, 0, 0,  0, 0, 1,  0, 0, 0],

                       [0, 9, 0,  8, 1, 0,  0, 0, 0],
                       [5, 3, 0,  0, 0, 0,  0, 4, 6],
                       [0, 0, 0,  0, 6, 5,  0, 3, 0],

                       [0, 0, 0,  1, 0, 0,  0, 0, 4],
                       [8, 0, 0,  0, 0, 9,  3, 0, 0],
                       [9, 4, 0,  5, 0, 0,  7, 0, 0]]

    elif puzzle_index == 2:
        init_config = [[2, 0, 0,  0, 0, 6,  0, 3, 0],
                       [4, 8, 0,  0, 1, 9,  0, 0, 0],
                       [0, 0, 7,  0, 2, 0,  9, 0, 0],

                       [0, 0, 0,  3, 0, 0,  0, 9, 0],
                       [7, 0, 8,  0, 0, 0,  1, 0, 5],
                       [0, 4, 0,  0, 0, 7,  0, 0, 0],

                       [0, 0, 4,  0, 9, 0,  6, 0, 0],
                       [0, 0, 0,  6, 4, 0,  0, 1, 9],
                       [0, 5, 0,  1, 0, 0,  0, 0, 8]]

    elif puzzle_index == 3:
        init_config = [[0, 0, 3,  2, 0, 0,  0, 7, 0],
                       [0, 0, 5,  0, 0, 0,  3, 0, 0],
                       [0, 0, 8,  9, 7, 0,  0, 5, 0],

                       [0, 0, 0,  8, 9, 0,  0, 0, 0],
                       [0, 5, 0,  0, 0, 0,  0, 2, 0],
                       [0, 0, 0,  0, 6, 1,  0, 0, 0],

                       [0, 1, 0,  0, 2, 5,  6, 0, 0],
                       [0, 0, 4,  0, 0, 0,  8, 0, 0],
                       [0, 9, 0,  0, 0, 7,  5, 0, 0]]

    elif puzzle_index == 4:
        init_config = [[0, 1, 0,  0, 0, 0,  0, 0, 2],
                       [8, 7, 0,  0, 0, 0,  5, 0, 4],
                       [5, 0, 2,  0, 0, 0,  0, 9, 0],

                       [0, 5, 0,  4, 0, 9,  0, 0, 1],
                       [0, 0, 0,  7, 3, 2,  0, 0, 0],
                       [9, 0, 0,  5, 0, 1,  0, 4, 0],

                       [0, 2, 0,  0, 0, 0,  4, 0, 8],
                       [4, 0, 6,  0, 0, 0,  0, 1, 3],
                       [1, 0, 0,  0, 0, 0,  0, 2, 0]]

    elif puzzle_index == 5:
        init_config = [[8, 9, 0,  2, 0, 0,  0, 7, 0],
                       [0, 0, 0,  0, 8, 0,  0, 0, 0],
                       [0, 4, 1,  0, 3, 0,  5, 0, 0],

                       [2, 5, 8,  0, 0, 0,  0, 0, 6],
                       [0, 0, 0,  0, 0, 0,  0, 0, 0],
                       [6, 0, 0,  0, 0, 0,  1, 4, 7],

                       [0, 0, 7,  0, 1, 0,  4, 3, 0],
                       [0, 0, 0,  0, 2, 0,  0, 0, 0],
                       [0, 2, 0,  0, 0, 7,  0, 5, 1]]

    elif puzzle_index == 6:
        # "World's hardest sudoku":
        # http://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html
        init_config = [[8, 0, 0,  0, 0, 0,  0, 0, 0],
                       [0, 0, 3,  6, 0, 0,  0, 0, 0],
                       [0, 7, 0,  0, 9, 0,  2, 0, 0],

                       [0, 5, 0,  0, 0, 7,  0, 0, 0],
                       [0, 0, 0,  0, 4, 5,  7, 0, 0],
                       [0, 0, 0,  1, 0, 0,  0, 3, 0],

                       [0, 0, 1,  0, 0, 0,  0, 6, 8],
                       [0, 0, 8,  5, 0, 0,  0, 1, 0],
                       [0, 9, 0,  0, 0, 0,  4, 0, 0]]

    elif puzzle_index == 7:
        init_config = [[1, 0, 0,  4, 0, 0,  0, 0, 0],
                       [7, 0, 0,  5, 0, 0,  6, 0, 3],
                       [0, 0, 0,  0, 3, 0,  4, 2, 0],

                       [0, 0, 9,  0, 0, 0,  0, 3, 5],
                       [0, 0, 0,  3, 0, 5,  0, 0, 0],
                       [6, 3, 0,  0, 0, 0,  1, 0, 0],

                       [0, 2, 6,  0, 5, 0,  0, 0, 0],
                       [9, 0, 4,  0, 0, 6,  0, 0, 7],
                       [0, 0, 0,  0, 0, 8,  0, 0, 2]]

    return np.array(init_config)





def validate_solution(solution):
    """validate a proposed solution for a sudoku field

    Args:
        solution (np.array): array of shape (9,9) containing integers between
        1 and 9

    Returns:
        (bool, np.array, np.array, np.array): tuple of values that indicate validity
        of the solution: 
        1. True if the overall solution is valid, False otherwise.
        2. boolean array of shape (3,3) that is True wherever a 3x3 box is valid
        3. boolean array of shape (9,) encoding the validity of all rows  
        4. boolean array of shape (9,) encoding the validity of all columns  
    """

    boxes = np.ones((3, 3), dtype=bool)
    rows = np.ones(9, dtype=bool)
    cols = np.ones(9, dtype=bool)

    expected_numbers = np.arange(1, 10)

    # validate boxes
    for i in range(3):
        for j in range(3):
            box = solution[3*i:3*i+3, 3*j:3*j+3]
            if len(np.setdiff1d(expected_numbers, box)) > 0:
                boxes[i, j] = False

    # validate rows and columns
    for i in range(9):
        if len(np.setdiff1d(expected_numbers, solution[i, :])) > 0:
            rows[i] = False
        if len(np.setdiff1d(expected_numbers, solution[:, i])) > 0:
            cols[i] = False

    # validate overall solution
    valid = boxes.all() and rows.all() and cols.all()

    return valid, boxes, rows, cols



























def plot_field(solution, with_color=False):
    background = np.full((200, 200, 3), 255, dtype=np.uint8) # white background
    field_ar = np.full((180, 180, 3), 255, dtype=np.uint8)

    #color in boxes in a checkerboard pattern
    for i in range(3):
        for j in range(3):
            if (i+j) % 2 == 0:
                field_ar[i*60:i*60+60, j*60:j*60+60] = dark_grey

    # draw the frames between cells in black
    for i in range(8):
        for j in range(8):
            field_ar[i*20+19:i*20+21,:] = black
            field_ar[:,j*20+19:j*20+21] = black

    field = Image.fromarray(field_ar)
    draw = ImageDraw.Draw(field)

    if with_color:
        valid, boxes, rows, cols = validate_solution(solution)        
        # color in rows and columns in the background
        for i in range(9):
            background[i*20+17:i*20+23,:] = green if rows[i] else red
            
            background[:,i*20+17:i*20+23] = green if cols[i] else red

        for i in range(9):
            for j in range(9):
                if solution[i,j] > 0:
                    color = green if boxes[i//3, j//3] else red
                    draw.text((j*20+4, i*20), str(solution[i, j]), color, font_digits)
    else:
        for i in range(9):
            for j in range(9):
                if solution[i,j] > 0:
                    draw.text((j*20+4, i*20), str(solution[i, j]), black, font_digits)

    # create outside frame for the sudoku field
    background[8:192,8:192,:] = 0

    background = Image.fromarray(background)
    background.paste(field, (10, 10))
    return background