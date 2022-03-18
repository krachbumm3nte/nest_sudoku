# %%

import sudoku_net
import sudoku_plotting
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
import os

# if __name__ == '__main__':

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%H:%M:%S')


def get_puzzle(puzzle_index):
    init_config = None

    if puzzle_index == 0:
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


# %%

net = sudoku_net.SudokuNet(n_digit=5)
max_sim_time = 10000
sim_time = 100
max_runs = max_sim_time//sim_time
max_repeats = 1
num_puzzles = 6
plot_outputs = False

noise_rates = np.arange(0,600,50)


performance = np.zeros((len(noise_rates), num_puzzles * max_repeats))

for index, rate in enumerate(noise_rates):
    net.set_noise_rate(rate)
    logging.info(f"repeating experiments with new noise frequency: {rate}")

    for puzzle in range(num_puzzles):

        net.reset()
        config = get_puzzle(puzzle)
        net.set_input_config(config)

        for repeat in range(max_repeats):
            logging.info(f"simulating rate {rate}, puzzle {puzzle}, rep {repeat}...")
            run = 0
            valid = False
            repeat_solution = 0

            net.reset_V_m()

            while not valid:
                net.reset_spike_recorders()
                net.run(sim_time)
                spikes = net.get_spike_trains()
                solution = np.zeros((9, 9))

                recorder_positions = net.stim_matrix

                for row in range(9):
                    for col in range(9):
                        positions = recorder_positions[row, col]

                        cell_spikes = spikes[positions]
                        rates = np.array([len(s["times"]) for s in cell_spikes])

                        winning_digit = int(np.random.choice(np.flatnonzero(rates == rates.max()))) + 1
                        solution[row,col] = winning_digit

                valid, cells, rows, cols = validate_solution(solution)
                
                if not valid:
                    ratio_correct = (np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27
                    logging.debug(f"performance: {ratio_correct}")
                else:
                    logging.info("valid solution found, simulation complete.")

                if plot_outputs:
                    img = sudoku_plotting.plot_field(solution, cells, rows, cols)
                    img.save(f"puzzle_{str(puzzle)}_rep_{str(repeat).zfill(2)}_{str(run).zfill(3)}.png")
                
                run += 1
                if run >= max_runs:
                    run = -1
                    logging.info(
                        f"no solution was found after {max_runs} iterations - aborting")
                    break
            
            performance[index, puzzle * max_repeats + repeat] = run
        
#%%
fig = plt.figure()

mean_durations = []

success = []

for puzzle in performance:
    foo = np.where(puzzle != -1.)
    mean_durations.append(puzzle[foo].mean()*sim_time)
    success.append(len(foo[0])/(max_repeats*num_puzzles))


fig, ax1 = plt.subplots()


ax1.plot(noise_rates, mean_durations, 'b:', label="mean convergence time")
ax1.set_ylabel("convergence time (ms)")
ax1.set_xticks(noise_rates)
ax1.set_ylim(0, 2000)
#ax.set_title("mean convergence time (ms)")

ax2 = ax1.twinx()
ax2.plot(noise_rates, success, 'r:', label = "likelihood of converging" )
ax2.set_ylabel("likelihood")

ax2.set_ylim(0, 1.1)
fig.legend()


# %%
plt.savefig("sudoku_convergence_1.png")
# %%
with open("results.pkl", "wb") as f:
    pickle.dump(performance, f)

# %%
