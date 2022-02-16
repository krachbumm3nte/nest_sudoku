import numpy as np
from PIL import Image, ImageDraw, ImageFont


# TODO: generalize this
font_loc = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
font_medium = ImageFont.truetype(font_loc, 18, encoding="unic")

green = (0, 128, 0)
red = (255, 0, 0)




def plot_field(solution, cells, rows, cols):
    background = np.zeros((200, 200, 3), dtype=np.uint8)


    field_ar = np.zeros((180, 180, 3), dtype=np.uint8)

    #color in boxes
    for i in range(3):
        for j in range(3):
            field_ar[i*60:i*60+60, j*60:j*60+60] = green if cells[i,j] else red

    # color in cells and columns
    for i in range(9):
        background[i*20+10:i*20+30, :] = green if rows[i] else red
           
        background[:, i*20+10:i*20+30] = green if cols[i] else red
            


    for i in range(8):
        for j in range(8):
            x_loc = i*20+19
            y_loc = j*20+19
            field_ar[x_loc:x_loc+2, :, :] = 255
            field_ar[:, y_loc:y_loc+2, :] = 255
    field = Image.fromarray(field_ar)
    draw = ImageDraw.Draw(field)

    for i in range(9):
        for j in range(9):
            draw.text((j*20+1, i*20+1),
                    str(int(solution[i, j])), (255, 255, 255), font_medium)

    background = Image.fromarray(background)
    background.paste(field, (10, 10))
    return background
