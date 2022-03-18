# -*- coding: utf-8 -*-
#
# sudoku_solver.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""Script controlling the simulation of a single game of Sudoku
----------------------------------------------------------------

This script instantiates a network and configures it to solve 
a sudoku puzzle. I then simulates the network until it either
converges on a solution or a maximum simulation time has been
reached.


[[ What kind of output is expected? ]]

The output shows the target neuron's membrane potential as a function of time.

See Also
---------
`Original implementation <https://github.com/SpiNNakerManchester/IntroLab/tree/master/sudoku>`_
:doc:`network class <sudoku_net.py>`

Notes
------
Terminology used in the documentation:
cell:       One of the 81 squares in the sudoku field
box:        one of the collections of 3x3 squares necessary to solve a Sudoku
digit:      Number between 1 and 9
population: collection of neurons coding for a digit in a single cell 

:Authors: J Gille
"""

import nest
import sudoku_net
import numpy as np
import logging
import pickle
from helpers import get_puzzle, validate_solution
nest.SetKernelStatus({'local_num_threads': 8})

puzzle_index = 3
noise_rate = 350
sim_time = 100
max_sim_time = 10000
max_iterations = max_sim_time//sim_time


puzzle = get_puzzle(puzzle_index)
network = sudoku_net.SudokuNet(n_digit=5, input=puzzle, noise_rate=noise_rate)
        
solution_states = np.zeros((max_iterations+1, 9, 9), dtype=int)
solution_states[0] = puzzle

run = 0
valid = False

while not valid:

    network.reset_spike_recorders()
    nest.Simulate(sim_time)

    spiketrains = network.get_spike_trains()
    solution = np.zeros((9, 9))

    for row in range(9):
        for col in range(9):
            spike_recorders = network.stim_matrix[row, col]

            cell_spikes = spiketrains[spike_recorders]
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
        logging.debug(f"run {run}, performance: {ratio_correct}")
    else:
        logging.info(f"run {run}, valid solution found.")
        break

    run += 1
    if run >= max_iterations:
        logging.info(
            f"no solution found after {max_iterations} iterations.")
        break

output = {}
output["noise_rate"] = noise_rate
output["sim_time"] = sim_time
output["max_sim_time"] = max_sim_time
output["solution_states"] = solution_states
output["puzzle"] = puzzle

with open(f"out/{noise_rate}Hz_puzzle_{puzzle}.pkl", "wb") as f:
    pickle.dump(output, f)
