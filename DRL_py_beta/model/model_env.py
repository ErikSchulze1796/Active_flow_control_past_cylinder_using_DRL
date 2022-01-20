import torch as pt
from nnEnvironmentModel import FFNN
from nnEnvironmentModel import WrapperModel

class modelEnv:
    
    def __init__(self, model_path, **kwargs) -> None:
        
        self.pmin = kwargs.get("pmin")
        self.pmax = kwargs.get("pmax")
        self.cdmin = kwargs.get("cdmin")
        self.cdmax = kwargs.get("cdmax")
        self.clmin = kwargs.get("clmin")
        self.clmax = kwargs.get("clmax")
        self.omegamin = kwargs.get("omegamin")
        self.omegamax = kwargs.get("omegamax")
        self.n_steps = kwargs.get("n_steps")
        self.n_sensors = kwargs.get("n_sensors")
        # Instantiate model and load trained parameters from directory
        self.model = FFNN(**kwargs)
        self.model.load_state_dict((pt.load(model_path)))
        # Instantiate wrapper model for min/max scaling on the fly
        self.wrapper = WrapperModel(self.model,
                                    self.pmin,
                                    self.pmax,
                                    self.omegamin,
                                    self.omegamax,
                                    self.cdmin,
                                    self.cdmax,
                                    self.clmin,
                                    self.clmax,
                                    self.n_steps,
                                    self.n_sensors)
        
    def get_prediction(self, input):
        return self.wrapper(input)
