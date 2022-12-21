

class StateMachine:

    def __init__(
        self,
        env,
        states,
        params={},
    ):
        self.env = env
        self.states = states # {"state name" : state transition function}
        self.current_state_name = None # current state name
        self.initial_state_name = None
        self.end_state_names = [] # list of possible final states (no state transition once one of these states are reached)
        self.params = params # initial parameters to pass into transition function (must be set if transition functions are parameterized)

    def add_state(self, name, state):
        """
        Adds new state to StateMachine

        Args:
            name: name of the state
            state (function): state machine state function
        """
        if name in self.states:
            raise Exception(f"State {name} already exists in StateMachine. Give another name")
        self.states[name] = state

    def set_initial_state(self, name):
        """
        Sets an initial state. State with "name" must be in StateMachine before setting initial state.

        Args:
            name: name of initial state (state with this name must exist in the StateMachine) 
        """
        if name not in self.states:
            raise Exception(f"State {name} does not exist. Add it during initialization or using add_state before calling this function.")
        
        self.initial_state_name = name
        self.current_state_name = name

    def set_end_state(self, name):
        """
        Sets a final state. State with "name" must be added to StateMachine with add_state before this funciton is called

        Args:
            name (str or list of str): name of end state (state with this name must exist in the StateMachine)
        """
        if isinstance(name, str):
            name = [name]

        for n in name:
            if n not in self.states:
                raise Exception(f"State {n} does not exist. Add it first using add_state.")

            if n not in self.end_state_names:
                self.end_state_names.append(n)

    def run(self):

        # make sure states is not empty, and initial and final states are set
        assert len(self.states) > 0, "StateMachine cannot be empty. Add states with add_state before running StateMachine"
        assert len(self.end_state_names) > 0, "At least one end state must exist in StateMachine before starting"
        assert self.initial_state_name is not None, "Initial state must be set with set_start_state"

        obs = self.env.reset()

        while True:
            
            # if state is one of end states, terminate
            if self.current_state_name in self.end_state_names:
                break
            
            # call transition function for current state
            transition_function = self.states[self.current_state_name]
            print(f"-------------------CALLED {self.current_state_name}")
            obs, next_state, params = transition_function(self.env, obs, self.params)
            self.current_state_name = next_state
            self.params = params


