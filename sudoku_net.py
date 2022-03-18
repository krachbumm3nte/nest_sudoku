import nest
import numpy as np
import logging

# TODO: contact hans eckehardt and/or dennis terhorst about licensing

nest.set_verbosity("M_WARNING")


inter_neuron_weight = -0.2
weight_stim = 1.
weight_noise = 1.6
delay = 1.0

neuron_params = {
    'C_m': 0.25,        # nF    membrane capacitance
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

    def __init__(self, n_digit=5, noise_rate=30., stim_rate=200., threads=10, input=None) -> None:

        nest.SetKernelStatus({'local_num_threads': threads})
        self.stim_rate = stim_rate
        self.n_digit = n_digit                  # number of neurons per digit
        self.n_cell = 9 * self.n_digit          # number of neurons per cell
        self.n_total = self.n_cell * 9 * 9      # total number of neurons
        # number of poisson generators for inputting a sudoku configuration
        self.n_stim_total = 9 ** 3

        logging.info("Creating neuron populations...")
        self.neurons = nest.Create('iaf_psc_exp', self.n_total,
                                   params=neuron_params)

        logging.info("Setting up noise...")
        self.noise_generator = nest.Create("poisson_generator", 1,
                                           {"rate": noise_rate})
        nest.Connect(self.noise_generator, self.neurons, 'all_to_all',
                     {'synapse_model': 'static_synapse', "delay": delay,
                      'weight': weight_noise})

        self.stim = nest.Create("poisson_generator", self.n_stim_total,
                                {'rate': self.stim_rate})
        self.spikerecorders = nest.Create('spike_recorder', self.n_stim_total)

        # matrix that stores NEST global_ids of the neurons in a structured way
        # for easy access during connection setup.
        # dimensions (in the sudoku field): [row, column, digit value, individual neuron]
        id_matrix = np.reshape(np.arange(self.n_total) + 1,
                               (9, 9, 9, self.n_digit))

        # Matrix that stores array indices (not global_ids) of stimulation sources
        # and spike recorders to be connected with the neurons.
        # dimensions: [row, column, digit value]
        self.stim_matrix = np.reshape(np.arange(self.n_stim_total), (9, 9, 9))

        """
        The functionality of the sudoku solver lies entirely within the inhibitory
        connections between neuron populations. each population of size :self.n_digit:
        encodes a digit between 1 and 9 in one of the 81 cells and has outgoing 
        inhibitory connections to several other populations. Namely all populations
        coding for the same digit in the same row, column and 3x3 box of the sudoku field,
        since a given digit can only ever occur once in any of those three. It also inhibits
        all populations in the same cell which code for different digits to force the network
        to converge on a single digit per cell. This behaviour is implemented in the following
        code segment.
        """
        logging.info("creating inhibitory connections between neurons...")
        for row in range(9):
            # A mask for indexing the id_matrix while excluding the current row
            row_mask = np.ones(9, dtype=bool)
            row_mask[row] = False

            # start and end row of the current 3x3 box
            box_row_start = (row // 3) * 3
            box_row_end = box_row_start + 3

            for column in range(9):
                # A mask for indexing the id_matrix while excluding the current column
                col_mask = np.ones(9, dtype=bool)
                col_mask[column] = False

                # start and end column of the current 3x3 box
                box_col_start = (column // 3) * 3
                box_col_end = box_col_start + 3

                # obtain the surrounding 3x3 box and remove the neurons of the current cell from it
                current_box = id_matrix[box_row_start:box_row_end,
                                        box_col_start:box_col_end]
                box_mask = np.ones((3, 3), dtype=bool)
                box_mask[row % 3, column % 3] = False
                current_box = current_box[box_mask]

                for digit in range(9):
                    digit_mask = np.ones(9, dtype=bool)
                    digit_mask[digit] = False

                    # all neurons coding for the current row, column and digit
                    sources = id_matrix[row, column, digit]

                    # all neurons in the same row coding for the same digit except those in the current cell
                    row_targets = id_matrix[row, col_mask, digit]
                    # same as above for the current column
                    col_targets = id_matrix[row_mask, column, digit]

                    # all neurons coding for different digits in the current cell
                    digit_targets = id_matrix[row, column, digit_mask]

                    # all neurons coding for the same digit in the current 3x3 box
                    box_targets = current_box[:, digit]

                    targets = np.concatenate((row_targets, col_targets, box_targets, digit_targets),
                                             axis=None)
                    # remove duplicates to avoid multapses
                    targets = np.unique(targets)

                    sources = nest.NodeCollection(sources)
                    targets = nest.NodeCollection(targets)
                    nest.Connect(sources, targets, 'all_to_all', {'synapse_model': 'static_synapse',
                                                                  "delay": delay, 'weight': inter_neuron_weight})

                    # connect the stimulation source to neurons at the current position
                    nest.Connect(self.stim[self.stim_matrix[row, column, digit]], sources,
                                 'all_to_all', {"delay": delay, "weight": 0.})

                    # connect all neurons at the current position to the same spike_recorder
                    nest.Connect(
                        sources, self.spikerecorders[self.stim_matrix[row, column, digit]])

        if input is not None:
            logging.info("setting input...")
            self.set_input_config(input, False)

        logging.info("Setup complete.")

    def reset_input(self):
        """sets all weights between input and network neurons to 0.
        """
        nest.GetConnections(self.stim).set({"weight": 0.})

    def set_input_config(self, input, clear_before=True):
        """sets the weight of the input generators according to a sudoku matrix.

        Args:
            input (np.array): a np.array of shape (9,9) where each entry is the value of the corresponding
            cell in the sudoku field.
            clear_before (bool, optional): reset input weights before setting the configuration. Defaults to True.
        """
        if clear_before:
            self.reset_input()

        for row in range(9):
            for column in range(9):
                val = input[row, column]
                if val != 0:
                    connections = nest.GetConnections(
                        self.stim[self.stim_matrix[row, column, val-1]])
                    connections.set({"weight": weight_stim})

    def run(self, runtime):
        """simulates the network for a given time.

        Args:
            runtime (int): total simulation time in ms.
        """
        nest.Simulate(runtime)

    def get_spike_trains(self):
        return np.array(self.spikerecorders.get("events"))

    def reset(self):
        """resets the network in three steps:
                setting all input weights to 0
                resetting all membrane potentials to their default value
                deleting all recorded spikes
        """
        self.reset_input()
        self.reset_V_m()
        self.reset_spike_recorders()

    def reset_V_m(self):
        """resets membrane potential of all neurons to (uniformly random) default values.
        """
        self.neurons.V_m = nest.random.uniform(-65, 55)

    def reset_spike_recorders(self):
        self.spikerecorders.n_events = 0

    def set_noise_rate(self, rate):
        """sets the rate of the poisson generator that feeds noise into the network.

        Args:
            rate (float): average spike frequency in Hz
        """
        self.noise_generator.rate = rate
