import argparse
import sudoku_net
import numpy as np
import logging
import pickle


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
            "Cannot return puzzle - index must be between 0 and 8!")

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
        2. boolean array of shape (3,3) that is True wherever a box is valid
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-rates", type=int, nargs='+',
                        default=[0, 350], help="noise rates (Hz) to simulate")
    parser.add_argument("-puzzles", type=int, nargs='+',
                        default=[0, 1, 2], help="puzzle ids to simulate for every noise rate")
    parser.add_argument("-max_time", type=int, default=10000,
                        help="maximum simulation time per configuration.")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%H:%M:%S')
    net = sudoku_net.SudokuNet(n_digit=5)

    sim_time = 100
    max_sim_time = args.max_time
    max_simulations = int(max_sim_time/sim_time)
    noise_rates = args.rates
    puzzles = args.puzzles

    for rate in noise_rates:
        net.set_noise_rate(rate)

        for puzzle in puzzles:
            config = get_puzzle(puzzle)
            net.set_input_config(config)
            logging.info(
                f"running simulation with frequency: {rate} on puzzle no. {puzzle}")

            net.reset()
            solution_states = np.zeros((max_simulations+1, 9, 9), dtype=int)

            solution_states[0] = config
            run = 0
            valid = False

            while not valid:

                net.reset_spike_recorders()
                net.run(sim_time)
                spiketrains = net.get_spike_trains()
                solution = np.zeros((9, 9))

                recorder_positions = net.stim_matrix

                for row in range(9):
                    for col in range(9):
                        positions = recorder_positions[row, col]

                        cell_spikes = spiketrains[positions]
                        noise_rates = np.array(
                            [len(s["times"]) for s in cell_spikes])

                        winning_digit = int(np.random.choice(
                            np.flatnonzero(noise_rates == noise_rates.max()))) + 1
                        solution[row, col] = winning_digit

                solution_states[run+1] = solution
                valid, cells, rows, cols = validate_solution(solution)

                if not valid:
                    ratio_correct = (
                        np.sum(cells) + np.sum(rows) + np.sum(cols)) / 27
                    logging.debug(f"performance: {ratio_correct}")
                else:
                    logging.info(f"valid solution found after {run} steps.")
                    break

                run += 1
                if run >= max_simulations:
                    logging.info(
                        f"no solution found after {max_simulations} iterations.")
                    break

            output = {}
            output["noise_rate"] = rate
            output["sim_time"] = sim_time
            output["max_sim_time"] = max_sim_time
            output["solution_states"] = solution_states
            output["puzzle"] = puzzle

            with open(f"out/{rate}Hz_puzzle_{puzzle}.pkl", "wb") as f:
                pickle.dump(output, f)
