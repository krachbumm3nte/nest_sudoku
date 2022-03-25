#%%
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pickle
import io
from helpers import validate_solution, plot_field
from glob import glob
import imageio

#%%

# final output image size
IMAGE_SIZE = np.array([420, 240])
background_color =  (255,255,255) # white
background_hex = "#ffffff"


dark_grey = (200, 200, 200)

black = (0,0,0)
white = (255,255,255)

# TODO: generalize this
font_loc = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
font_text = ImageFont.truetype(font_loc, 14, encoding="unic")

green = (0, 105, 0)
red = (150, 0, 0)

in_folder = "out"

#%%

image_count = 0


performance = []

styles = ["r.", "b.", "go"]
colors = [(1., 0.4, 0.), (0., 0.2, 0.6), (0., 0.2, 0.6)]

#for file_index, file in enumerate(os.listdir("out")):

files = ["10Hz_puzzle_3.pkl", "350Hz_puzzle_3.pkl", "350Hz_puzzle_4.pkl"]
performance = []

lines = [None, None, None]
plt.clf()
# plot performance of the current run
fig = plt.figure(facecolor=background_hex)
ax = plt.axes()
ax.set_facecolor(background_hex)
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 8
plt.ylabel("performance")
plt.xlabel("simulation time (ms)")
plt.xticks([0, 50, 100], ["0", "5e3", "1e4"])
plt.yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])
ax.set_xlim(0,100)
ax.set_ylim(0,1)
plt.plot([1,1], "wo")
for file_index, file in enumerate(files):
    print(file)
    with open("out/" + file, "rb") as f:
        simulation_data = pickle.load(f)

    current_run = simulation_data["solution_states"]
    noise_rate = simulation_data["noise_rate"]
    puzzle = simulation_data["puzzle"]
    performance.append({"noise_rate" :noise_rate, "color": colors[file_index], "data": []})

    for i in range(len(current_run) + 1):
        solution = current_run[i-1]

        background = Image.new("RGB", tuple(IMAGE_SIZE), background_color)
        draw = ImageDraw.Draw(background)

        if i == 0:
            field = plot_field(puzzle, puzzle, False)
            performance[file_index]["data"].append(0.)
            image_repeat = 8
        else:
            valid, cells, rows, cols = validate_solution(puzzle, solution)
            field = plot_field(puzzle, solution, True)
            performance[file_index]["data"].append((np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27)
            image_repeat = 1
        
        background.paste(field, (215, 5))
        
        sim_time = i * simulation_data["sim_time"]

        plt.clf()
        # plot performance of the current run
        fig = plt.figure(facecolor=background_hex)
        ax = plt.axes()
        ax.set_facecolor(background_hex)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 8
        plt.ylabel("performance")
        plt.xlabel("simulation time (ms)")
        plt.xticks([0, 50, 100], ["0", "5e3", "1e4"])
        plt.yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])
        ax.set_xlim(0,100)
        ax.set_ylim(0,1)

        # set a constant figsize by pixel size
        DPI = fig.get_dpi()
        fig.set_size_inches(215.0/DPI,160.0/DPI)
        for p in performance:
            r = p["noise_rate"]
            lines[file_index] = ax.plot(p["data"], color=p["color"], label=f"{r}Hz")[0]
        if file_index == 0:
            ax.legend(handles = [lines[0]], loc="lower right")
        else:
            ax.legend(handles = [lines[0], lines[1]], loc="lower right")
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


filenames = sorted(glob(os.path.join("imgs", "*.png")))
out_file = "foo.gif"
with imageio.get_writer(out_file, mode='I', fps=4) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print(f"gif created under: {out_file}")
# %%
