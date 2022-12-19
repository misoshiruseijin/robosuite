

class StateMachine:

    def __init__(self, env):
        
        self.states = {} # {"state name" : state function}
        self.current_state_name = None # current state name
        self.initial_state_name = None
        self.end_state_names = [] # list of possible final states (no state transition once one of these states are reached)
        # self.ready = False # whether robot is ready to move on to next state
        self.use_params = True # set to True if transition function require parameters

        if self.use_params:
            self.params = {}

        # TODO - need initial observation?

    def add_state(self, name, state):
        """
        Adds new state to StateMachine

        Args:
            name: name of the state
            state (function): state machine state function
        """
        self.states[name] = state

    def set_start_state(self, name, params):
        """
        Sets an initial state. State with "name" must be added to StateMachine with add_state before this function is called

        Args:
            name: name of initial state (state with this name must exist in the StateMachine) 
        """
        if name not in self.states:
            raise Exception(f"State {name} does not exist. Add it first using add_state.")
        
        self.initial_state_name = name
        self.current_state_name = name
        if self.use_params:
            self.params = params

    def set_end_state(self, name):
        """
        Sets a final state. State with "name" must be added to StateMachine with add_state before this funciton is called

        Args:
            name: name of end state (state with this name must exist in the StateMachine)
        """
        if name not in self.states:
            raise Exception(f"State {name} does not exist. Add it first using add_state.")

        if name not in self.end_state_names:
            self.end_state_names.append(name)

    def run(self):

        # make sure states is not empty, and initial and final states are set
        assert len(self.states) > 0, "StateMachine cannot be empty. Add states with add_state before running StateMachine"
        assert len(self.end_state_names) > 0, "At least one end state must exist in StateMachine before starting"
        assert self.initial_state_name is not None, "Initial state must be set with set_start_state"

        while True:
            # call transition function for current state
            transition_function = self.states[self.current_state_name]
            transition_function()
            # if state is one of end states, terminate

