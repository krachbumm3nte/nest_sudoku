# %%

import sudoku_net
import sudoku_plotting
import matplotlib.pyplot as plt
import numpy as np
import logging

# if __name__ == '__main__':

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%H:%M:%S')

puzzle = None

init_config = None

if puzzle == 1:
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
elif puzzle == 2:
    init_config = [[2, 0, 0,  0, 0, 6,  0, 3, 0],
                   [4, 8, 0,  0, 1, 9,  0, 0, 0],
                   [0, 0, 7,  0, 2, 0,  9, 0, 0],

                   [0, 0, 0,  3, 0, 0,  0, 9, 0],
                   [7, 0, 8,  0, 0, 0,  1, 0, 5],
                   [0, 4, 0,  0, 0, 7,  0, 0, 0],

                   [0, 0, 4,  0, 9, 0,  6, 0, 0],
                   [0, 0, 0,  6, 4, 0,  0, 1, 9],
                   [0, 5, 0,  1, 0, 0,  0, 0, 8]]
elif puzzle == 3:
    init_config = [[0, 0, 3,  2, 0, 0,  0, 7, 0],
                   [0, 0, 5,  0, 0, 0,  3, 0, 0],
                   [0, 0, 8,  9, 7, 0,  0, 5, 0],

                   [0, 0, 0,  8, 9, 0,  0, 0, 0],
                   [0, 5, 0,  0, 0, 0,  0, 2, 0],
                   [0, 0, 0,  0, 6, 1,  0, 0, 0],

                   [0, 1, 0,  0, 2, 5,  6, 0, 0],
                   [0, 0, 4,  0, 0, 0,  8, 0, 0],
                   [0, 9, 0,  0, 0, 7,  5, 0, 0]]
elif puzzle == 4:
    init_config = [[0, 1, 0,  0, 0, 0,  0, 0, 2],
                   [8, 7, 0,  0, 0, 0,  5, 0, 4],
                   [5, 0, 2,  0, 0, 0,  0, 9, 0],

                   [0, 5, 0,  4, 0, 9,  0, 0, 1],
                   [0, 0, 0,  7, 3, 2,  0, 0, 0],
                   [9, 0, 0,  5, 0, 1,  0, 4, 0],

                   [0, 2, 0,  0, 0, 0,  4, 0, 8],
                   [4, 0, 6,  0, 0, 0,  0, 1, 3],
                   [1, 0, 0,  0, 0, 0,  0, 2, 0]]
elif puzzle == 5:
    init_config = [[8, 9, 0,  2, 0, 0,  0, 7, 0],
                   [0, 0, 0,  0, 8, 0,  0, 0, 0],
                   [0, 4, 1,  0, 3, 0,  5, 0, 0],

                   [2, 5, 8,  0, 0, 0,  0, 0, 6],
                   [0, 0, 0,  0, 0, 0,  0, 0, 0],
                   [6, 0, 0,  0, 0, 0,  1, 4, 7],

                   [0, 0, 7,  0, 1, 0,  4, 3, 0],
                   [0, 0, 0,  0, 2, 0,  0, 0, 0],
                   [0, 2, 0,  0, 0, 7,  0, 5, 1]]
elif puzzle == 6:
    # "World's hardest sudoku":
    # http://www.telegraph.co.uk/news/science/science-news/9359579/\
    # Worlds-hardest-sudoku-can-you-crack-it.html
    init_config = [[8, 0, 0,  0, 0, 0,  0, 0, 0],
                   [0, 0, 3,  6, 0, 0,  0, 0, 0],
                   [0, 7, 0,  0, 9, 0,  2, 0, 0],

                   [0, 5, 0,  0, 0, 7,  0, 0, 0],
                   [0, 0, 0,  0, 4, 5,  7, 0, 0],
                   [0, 0, 0,  1, 0, 0,  0, 3, 0],

                   [0, 0, 1,  0, 0, 0,  0, 6, 8],
                   [0, 0, 8,  5, 0, 0,  0, 1, 0],
                   [0, 9, 0,  0, 0, 0,  4, 0, 0]]
elif puzzle == 7:
    init_config = [[1, 0, 0,  4, 0, 0,  0, 0, 0],
                   [7, 0, 0,  5, 0, 0,  6, 0, 3],
                   [0, 0, 0,  0, 3, 0,  4, 2, 0],

                   [0, 0, 9,  0, 0, 0,  0, 3, 5],
                   [0, 0, 0,  3, 0, 5,  0, 0, 0],
                   [6, 3, 0,  0, 0, 0,  1, 0, 0],

                   [0, 2, 6,  0, 5, 0,  0, 0, 0],
                   [9, 0, 4,  0, 0, 6,  0, 0, 7],
                   [0, 0, 0,  0, 0, 8,  0, 0, 2]]


# %%

net = sudoku_net.SudokuNet(n_digit=5, n_stim=30, input=init_config)


#%%
def validate_solution(matrix):

    cells = np.ones((3,3), dtype=bool)
    rows = np.ones(9, dtype=bool)
    cols = np.ones(9, dtype=bool)
    
    expected_numbers = np.arange(1, 10)


    for i in range(3):
        for j in range(3):
            box = matrix[3*i:3*i+3, 3*j:3*j+3]
            if len(np.setdiff1d(expected_numbers, box)) > 0:
                cells[i,j] = False
    


    for i in range(9):
        if len(np.setdiff1d(expected_numbers, matrix[i, :])) > 0:
            rows[i] = False
        if len(np.setdiff1d(expected_numbers, matrix[:, i])) > 0:
            cols[i] = False

    valid = cells.all() and rows.all() and cols.all()

    return valid, cells, rows, cols


# %%

run = 0
valid = False
previous_solution = None
repeat_solution = 0

net.reset_network()

while not valid:
    net.reset_spike_recorders()
    net.run(400)
    spikes = net.get_spikes()

    solution = np.zeros((9, 9))

    for y in range(9):
        for x in range(9):
            offset = ((y * 9) + x) * net.n_cell

            cell_spikes = spikes[offset:offset + net.n_cell]
            rates = [len(s["times"]) for s in cell_spikes]
            out = []
            for i in range(0, len(cell_spikes), net.n_digit):
                out.append(sum(rates[i:i+net.n_digit]))

            m = max(out)
            solution[8 - y][x] = out.index(m) + 1

    if solution is previous_solution:
        repeat_solution += 1
    else:
        repeat_solution = 0
    previous_solution = solution

    if repeat_solution >= 5:
        logging.info(f"stuck in a local optimum for 5 iterations, aborting")
        break

    valid, cells, rows, cols = validate_solution(solution)
    
    ratio_correct = (np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27

    logging.info(f"performance: {ratio_correct}")

    img = sudoku_plotting.plot_field(solution, cells, rows, cols)
    img.save(f"foo_{str(run).zfill(3)}.png")
    run += 1


# %%
