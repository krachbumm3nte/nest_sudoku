#%%
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pickle
import io


# final output image size
IMAGE_SIZE = np.array([400, 240])
background_color =  (255,255,255) # white
background_hex = "#ffffff"


dark_grey = (200, 200, 200)

black = (0,0,0)
white = (255,255,255)

# TODO: generalize this
font_loc = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
font_digits = ImageFont.truetype(font_loc, 18, encoding="unic")
font_text = ImageFont.truetype(font_loc, 14, encoding="unic")

green = (0, 105, 0)
red = (150, 0, 0)

in_folder = "out"


def plot_field(solution, cells=None, rows=None, cols=None):
    background = np.full((200, 200, 3), 255, dtype=np.uint8) # white background
    field_ar = np.full((180, 180, 3), 255, dtype=np.uint8)

    if cells is not None:        
        # color in cells and columns
        for i in range(9):
            background[i*20+17:i*20+23,:] = green if rows[i] else red
            
            background[:,i*20+17:i*20+23] = green if cols[i] else red
            
    background[8:192,8:192:] = 0

    #color in boxes
    for i in range(3):
        for j in range(3):
            if (i+j) % 2 == 0:
                field_ar[i*60:i*60+60, j*60:j*60+60] = dark_grey


    for i in range(8):
        for j in range(8):
            x_loc = i*20+19
            y_loc = j*20+19
            field_ar[x_loc:x_loc+2,:] = black
            field_ar[:,y_loc:y_loc+2] = black
    field = Image.fromarray(field_ar)
    draw = ImageDraw.Draw(field)

    for i in range(9):
        for j in range(9):
            if solution[i,j] > 0:
                if cells is not None:
                    color = green if cells[i//3, j//3] else red
                else:
                    color = black
                draw.text((j*20+4, i*20),
                    str(solution[i, j]), color, font_digits)

    background = Image.fromarray(background)
    background.paste(field, (10, 10))
    return background

def validate_solution(matrix):

    cells = np.ones((3, 3), dtype=bool)
    rows = np.ones(9, dtype=bool)
    cols = np.ones(9, dtype=bool)

    expected_numbers = np.arange(1, 10)

    for i in range(3):
        for j in range(3):
            box = matrix[3*i:3*i+3, 3*j:3*j+3]
            if len(np.setdiff1d(expected_numbers, box)) > 0:
                cells[i, j] = False

    for i in range(9):
        if len(np.setdiff1d(expected_numbers, matrix[i, :])) > 0:
            rows[i] = False
        if len(np.setdiff1d(expected_numbers, matrix[:, i])) > 0:
            cols[i] = False

    valid = cells.all() and rows.all() and cols.all()

    return valid, cells, rows, cols

#%%
"""
with open("results.pkl", "rb") as f:
    long_time_sim = np.array(pickle.load(f))
convergence_likelihood = [sum(x > 0) / long_time_sim.shape[1] for x in long_time_sim]
"""

image_count = 0


performance = []

styles = ["r.", "b.", "go"]
colors = [(1., 0.4, 0.), (0., 0.2, 0.6), (0., 0.2, 0.6)]






#for file_index, file in enumerate(os.listdir("out")):

files = ["10Hz_puzzle_2_00.pkl", "350Hz_puzzle_2_00.pkl", "350Hz_puzzle_1_02.pkl"]
performance = []


for file_index, file in enumerate(files):
    print(file)
    with open("out/" + file, "rb") as f:
        simulation_data = pickle.load(f)

    current_run = simulation_data["solution_states"]
    noise_rate = simulation_data["noise_rate"]

    performance.append({"noise_rate" :noise_rate, "color": colors[file_index], "data": []})

    for i in range(len(current_run)):
        solution = current_run[i]
        if solution.sum() == 0:
            break


        background = Image.new("RGB", tuple(IMAGE_SIZE), background_color)
        draw = ImageDraw.Draw(background)

        if i == 0:
            field = plot_field(solution)
            performance[file_index]["data"].append(0.)
            image_repeat = 8
        else:
            valid, cells, rows, cols = validate_solution(solution)
            field = plot_field(solution, cells, rows, cols)
            performance[file_index]["data"].append((np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27)
            image_repeat = 1
        
        background.paste(field, (195, 5))
        
        sim_time = i * simulation_data["sim_time"]

        plt.clf()
        # plot performance of the current run
        fig = plt.figure(facecolor=background_hex)
        ax = plt.axes()
        ax.set_facecolor(background_hex)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 10
        plt.ylabel("performance")
        plt.xlabel("simulation time (ms)")
        plt.xticks([0, 50, 100], ["0", "5e3", "1e4"])
        plt.yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])
        ax.set_xlim(0,100)
        ax.set_ylim(0,1)

        # set a constant figsize by pixel size
        DPI = fig.get_dpi()
        fig.set_size_inches(195.0/DPI,160.0/DPI)
        for p in performance:
            r = p["noise_rate"]
            ax.plot(p["data"], color=p["color"], label=f"{r}Hz")
        ax.legend(loc="lower right")
        # fishy workaround to turn a pyplot figure into a PIL.Image
        buf = io.BytesIO()
        fig.savefig(buf)
        plt.close(fig)
        buf.seek(0)
        performance_plot = Image.open(buf)
        background.paste(performance_plot, (0, 65))


        draw.text((5, 5), "Noise rate:", black, font_text)
        draw.text((5, 20), f"{noise_rate}Hz", black, font_text)

        draw.text((5, 40), f"Simulation time:", black, font_text)
        draw.text((5, 55), f"{sim_time}ms", black, font_text)


        for j in range(image_repeat):
            background.save(f"imgs/{str(file_index).zfill(3)}_{str(image_count).zfill(3)}.png")
            image_count += 1

    for j in range(15):
        background.save(f"imgs/{str(file_index).zfill(3)}_{str(image_count).zfill(3)}.png")
        image_count += 1

# %%
