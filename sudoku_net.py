import nest
import numpy as np
import logging

# TODO: contact hans eckehardt and/or dennis terhorst about licensing

nest.set_verbosity("M_WARNING")


weight_cell = -0.2
weight_stim = 1.
weight_nois = 1.6
delay = 2.0

neuron_params = {
    'C_m': 0.25,        # nF membrane capacitance
    'I_e': 0.5,         # nA    bias current
    'tau_m': 20.0,      # ms    membrane time constant
    't_ref': 2.0,       # ms    refractory period
    'tau_syn_ex': 5.0,  # ms    excitatory synapse time constant
    'tau_syn_in': 5.0,  # ms    inhibitory synapse time constant
    'V_reset': -70.0,   # mV    reset membrane potential
    'E_L': -65.0,       # mV    resting membrane potential
    'V_th': -50.0,      # mV    firing threshold voltage
    'V_m': nest.random.uniform(min=-65, max=-55)
}


class SudokuNet:

    def __init__(self, n_digit = 5, n_stim = 30, input=None) -> None:

        nest.SetKernelStatus({'local_num_threads': 8})

        self.n_digit = n_digit                   # number of neurons per digit
        self.n_cell = 9 * self.n_digit
        self.n_stim = n_stim
        self.n_row = self.n_cell * 9
        self.n_total = self.n_row * 9
        self.n_stim_total = self.n_stim * 9 * 9

        self.cells = nest.Create('iaf_psc_exp', self.n_total, params=neuron_params)
        self.spikerecorder = nest.Create('spike_recorder', self.n_total)
        nest.Connect(self.cells, self.spikerecorder, {"rule": "one_to_one"})

        logging.info("Creating Noise Sources...")
        self.noise = nest.Create("poisson_generator", self.n_total, {"rate": 20.0})
        nest.Connect(self.noise, self.cells, 'one_to_one', {
                    'synapse_model': 'static_synapse', "delay": delay, 'weight': weight_nois})


        """
        logging.info("Setting up inhibition between digits in a cell...")
        weight_mat = np.ones((9, 9))
        for i in range(9):
            weight_mat[i, i] = 0
        weight_mat = np.kron(weight_mat, np.ones((self.n_digit, self.n_digit)))

        for column in range(9):
            for row in range(9):
                # base neuron offset over every cell (x,y)
                position_offset = ((row * 9) + column) * self.n_cell + 1

                # list of nodecollections each representing a digit of the current cell (x, y)
                digit_populations = [nest.NodeCollection(np.arange(position_offset + i * self.n_digit, position_offset + (i + 1) * self.n_digit)) for i in range(9)]

                for i in range(9):
                    for j in range(9):
                        # connect all populations to all others, but not to themselves
                        if i != j:
                            nest.Connect(digit_populations[i], digit_populations[j], 'all_to_all', {
                                        'synapse_model': 'static_synapse', "delay": delay, 'weight': weight_cell})

        
        logging.info("Setting up inhibition between cells...")
        for x in range(9):
            for y in range(9):
                for r in range(9):
                    if r != x:
                        self.interCell(x, y, r, y)  # by row...
                for c in range(9):
                    if c != y:
                        self.interCell(x, y, x, c)  # by column...
                for r in range(3 * (x // 3), 3 * (x // 3 + 1)):
                    for c in range(3 * (y // 3), 3 * (y // 3 + 1)):
                        if r != x and c != y:
                            self.interCell(x, y, r, c)  # & by square     



        base_row_ids = np.array([i for i in range(self.n_row) if i//self.n_digit%9 == 0])
        # neuron ids coding for number 1 in all cells across the first row
        # with default params: [0, 1, 2, 3, 4, 45, 46, 47, 48, 49, 90, 91, 92, 93, ..., 363, 364]
        # this basic pattern will be transformed and multiplied to adress the neuron populations 
        # of each cell and digit.

        for digit in range(9):
            # offset to the base indices for the current digit. +1 because global_id in NEST starts counting at 1
            digit_offset = digit * self.n_digit + 1


            rows = np.array([base_row_ids + digit_offset + (i * self.n_row) for i in range(9)])

            for column in range(9):
                column_indices = np.arange(column*n_digit, (column+1)*n_digit)

                current_column = rows[:,column_indices]
                target_columns = np.delete(rows, column_indices, 1)
                
                col_start = column//3 * self.n_digit * 3
                col_end = col_start + self.n_digit * 3

                for row in range(9):
                    sources = current_column[row]

                    current_column_targets = np.delete(current_column, row, 0).flatten()
                    row_target = target_columns[row]

                    row_start = row//3 * 3
                    row_end = row_start + 3

                    current_box = rows[row_start:row_end,col_start:col_end]
                    box_mask = np.ones((3, 3*n_digit), dtype=bool)
                    box_mask[row%3,column%3*self.n_digit:(column%3+1)*self.n_digit] = False
                    box_target = current_box[box_mask]

                    targets= np.concatenate((row_target, current_column_targets, box_target))
                    targets = np.unique(targets)

                    if 91 in sources:
                        print("foo")

                    sources = nest.NodeCollection(sources)
                    targets = nest.NodeCollection(targets)
                    nest.Connect(sources, targets, 'all_to_all',
                            {'synapse_model': 'static_synapse', "delay": delay, 'weight': weight_cell})
        """       

        id_matrix = np.reshape(np.arange(self.n_total)+1, (9,9,9,self.n_digit))
        
        for column in range(9):
            # A mask for indexing the id_matrix while excluding the current column
            col_mask = np.ones(9, dtype=bool)
            col_mask[column] = False

            # start and end column of the current 3x3 box
            box_col_start = (column // 3) * 3
            box_col_end = box_col_start + 3

            for row in range(9):
                # A mask for indexing the id_matrix while excluding the current row
                row_mask = np.ones(9, dtype=bool)
                row_mask[row] = False
                
                # start and end row of the current 3x3 box
                box_row_start = (row // 3) * 3
                box_row_end = box_row_start + 3

                # obtain the 3x3 box and remove the neurons of the current cell from it
                current_box = id_matrix[box_row_start:box_row_end, box_col_start:box_col_end]
                box_mask = np.ones((3, 3), dtype=bool)
                box_mask[row%3,column%3] = False
                current_box = current_box[box_mask]
        
                for digit in range(9):
                    digit_mask = np.ones(9, dtype=bool)
                    digit_mask[digit] = False

                    # all neurons coding for the current row, column and digit
                    sources = id_matrix[row, column, digit]

                    # all neurons in the same row coding for the same digit except those in the current cell
                    row_targets = id_matrix[row, col_mask, digit].flatten()
                    # same as above for the current column
                    col_targets = id_matrix[row_mask, column, digit].flatten()
                    
                    # all neurons coding for different digits in the current cell
                    digit_targets = id_matrix[row, column, digit_mask].flatten()

                    # all neurons coding for the same digit in the current 3x3 box
                    box_targets = current_box[:,digit].flatten()

                    targets = np.concatenate((row_targets, col_targets, box_targets, digit_targets))
                    targets = np.unique(targets) # remove duplicates to avoid multapses


                    sources = nest.NodeCollection(sources)
                    targets = nest.NodeCollection(targets)
                    nest.Connect(sources, targets, 'all_to_all',
                            {'synapse_model': 'static_synapse', "delay": delay, 'weight': weight_cell})
        

        self.stim = nest.Create("poisson_generator", self.n_stim_total, {'rate': 10.})

        if input is not None:
            self.set_stim_connections(input)



    def interCell(self, x, y, r, c):
        """ Inhibit same number: connections are n_N squares on diagonal of
            weight_cell() from cell[x][y] to cell[r][c]
        """

        base_source = ((y * 9) + x) * self.n_cell
        base_dest = ((c * 9) + r) * self.n_cell

        for i in range(self.n_cell):
            for j in range(self.n_digit * (i // self.n_digit), self.n_digit * (i // self.n_digit + 1)):
                nest.Connect(self.cells[i + base_source], self.cells[j + base_dest], 'one_to_one',
                            {'synapse_model': 'static_synapse', "delay": delay, 'weight': weight_cell})



    def reset_stim_connections(self):
        nest.GetConnections(self.stim).set({"weight": 0.})

    def set_stim_connections(self, input, clear_before = True):

        for x in range(9):
            for y in range(9):
                if input[8 - y][x] != 0:
                    base_stim = ((y * 9) + x) * self.n_stim
                    base = ((y * 9) + x) * self.n_cell
                    for i in range(self.n_stim):

                        # one n_N square on diagonal
                        for j in range(self.n_digit * (input[8 - y][x] - 1), self.n_digit * input[8 - y][x]):
                            nest.Connect(self.stim[i + base_stim], self.cells[j + base],
                                        'one_to_one', {"delay": delay, "weight": weight_stim})

    def run(self, runtime):
        logging.info(f"simulating network for {runtime}ms..")
        nest.Simulate(runtime)
        logging.info("Simulation complete")

    def get_spikes(self):
        return self.spikerecorder.get("events")

    def reset_network(self):
        self.cells.V_m=nest.random.uniform(-65, 55)

    def reset_spike_recorders(self):
        self.spikerecorder.n_events = 0