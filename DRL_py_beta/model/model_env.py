import torch as pt
from model.nnEnvironmentModel import FFNN
from model.nnEnvironmentModel import WrapperModel

class modelEnv:
    """Model environment class for use in DRL
    """
    
    def __init__(self, model_path, **kwargs) -> None:
        """Constructor of modelEnv class

        Parameters
        ----------
        model_path : string
            Path to environment model
        """
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
        # self.model.load_state_dict((pt.load(model_path)))
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
        self.wrapper.load_state_dict((pt.load(model_path)))
        
    def get_prediction(self, input):
        """Returns a prediction using the environment surrogate model

        Parameters
        ----------
        input : torch.Tensor
            Combined input feature tensor for using with the FFNN

        Returns
        -------
        torch.Tensor
            Next predicted state
        """
        return self.wrapper(pt.from_numpy(input))
