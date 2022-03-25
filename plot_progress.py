import argparse
import os
import sys
import pickle
import imageio
from glob import glob
from helpers import plot_field

temp_dir = "tmp"
out_file = "sudoku.gif"
in_file = "350Hz_puzzle_3.pkl"
keep_temps = True

try:
    os.mkdir(temp_dir)
except:
    print(f"temporary file folder ({temp_dir}) already exists! Aborting.")
    #sys.exit()

if os.path.exists(out_file):
    print(f"Target file ({out_file}) already exists! Aborting.")
    #sys.exit()

with open(in_file, "rb") as f:
    simulation_data = pickle.load(f)

solution_states = simulation_data["solution_states"]

image_count = 0


for i in range(8):
    field = plot_field(simulation_data['puzzle'], simulation_data['puzzle'], False)


for i in range(len(solution_states)):
    current_state = solution_states[i]

    if i == 0:
        # repeat the (colorless) starting configuration several times
        image_repeat = 8
    else:
        field = plot_field(simulation_data['puzzle'], current_state, True)
        image_repeat = 1

    if i == len(solution_states) - 1:
        # repeat the final solution a few more times to make it observable
        # before the gif loops again
        image_repeat = 15

    for j in range(image_repeat):
        field.save(os.path.join(temp_dir, f"{str(image_count).zfill(4)}.png"))
        image_count += 1

filenames = sorted(glob(os.path.join(temp_dir, "*.png")))

with imageio.get_writer(out_file, mode='I', fps=4) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print(f"gif created under: {out_file}")

if not keep_temps:
    print("deleting temporary image files...")
    for in_file in filenames:
        os.unlink(in_file)
    os.rmdir(temp_dir)
