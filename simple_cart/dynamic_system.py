class DynamicSystem():
    def __init__(self, state, gain, transfer, static_parameters):
        self.state = state
        self.gain = gain
        self.transfer = transfer
        self.static_parameters = static_parameters

        self.state_shape = len(state)
        self.gain_shape = len(gain)
