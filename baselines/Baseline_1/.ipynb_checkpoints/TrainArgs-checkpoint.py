"""
Dataclass to hold training args.
Conforms to accelerate's `state_dict` and
`load_state_dict` protocol for saving.

@dataclass general reference:
https://peps.python.org/pep-0557/
"""

from dataclasses import dataclass
import os
import time
import datetime

@dataclass
class TrainArgs:
    
    MODEL_NAME: str = 'vae'

    # data splitting seeds
    CV_SEED_GB3 = 243858
    CV_SEED_BPTI = 647899
    CV_SEED_UBIQ = 187349
    CV_SEED_1bxy = 133538
    CV_SEED_1bx7 = 988573
    CV_SEED_1ptq = 781593

    # VAE-specific: model dimensions from Tian et al. 2011
    # VAE_INPUT_DIM: int = 168 # varies, set programmatically
    ENCODER_DIM_ARR = [1024, 256, 64]
    LATENT_SPACE_DIMS = [16, 2]
    DECODER_DIM_ARR = [16, 64, 256, 1024]
    
    # hardware: changed if nec. in __post_init__() below
    MACHINE: str = 'local'
    ON_CPU: bool = True
    DATALOADER_N_WORKERS: int = 0
    DATALOADER_DROP_LAST: bool = False
    
    # training params
        # these can be overwritten by command line args if coded as such!
        # burn-in num of epochs prevents model saving until its reached
        # change batch size depending on memory avail; use powers of 2
    SAVE_FINAL_MODEL: bool = True
    VALID_PROP: float = 0.21 # 0.20 may under-populate the valid set
    TVT_SPLIT_RS: int = 548708
    DATALOADER_RS: int = 437589
    BURNIN_N_EPOCHS: int = 5
    N_EPOCHS: int = 50
    BATCH_SIZE: int = 256
    LEARN_RATE: float = 0.001
    ADAM_BETAS = (0.9, 0.999)
    RELU_NSLOPE = 0.2

    # args for saving 'best' model during training, by a 
    # validation metric
    NO_VALID_LOSS_IMPROVE_PATIENCE: int = 32
    MAIN_METRIC: str = 'loss_valid'
    MAIN_METRIC_IS_BETTER: str = 'lower' # or: 'higher'
    MAIN_METRIC_REL_IMPROV_THRESH: float = 0.999

    # paths inits (actually set in __post_init__ below)
    ROOT: str = "../"
    DATA_DIR: str = None
    MODEL_SAVE_DIR: str = None
    PRINT_DIR: str = None
    TRAIN_LOGS_SAVE_DIR: str = None
    MODEL_SAVE_SUBDIR: str = None
    
    def __post_init__(self):
        """
        Change directory paths, depending on the machine in use.
        """
        # use GM time for timestamping: otherwise running on machines
        # in different time zones will cause inconsistent times!
        ts = time.gmtime()
        ts = time.strftime('%Y-%m-%d-%H-%M', ts)
        model_ts = f"{self.MODEL_NAME}_{ts}"
        
        if self.MACHINE in ('local', 'Local'):
            """
            NOTE: here you can change the dirs based on a passed 'machine' clarg
            """
            self.ROOT = "../"

        self.DATA_DIR = f"{self.ROOT}/data"
        self.DATA_TRANSFORM_INFO_DIR = f"{self.DATA_DIR}"
        if self.MODEL_SAVE_SUBDIR is not None:
            self.MODEL_SAVE_DIR = f"{self.ROOT}/models/{self.MODEL_SAVE_SUBDIR}/{model_ts}"
        else: 
            self.MODEL_SAVE_DIR = f"{self.ROOT}/models/{model_ts}"
        self.PRINT_DIR = f"{self.MODEL_SAVE_DIR}/out.txt" # print output file saved w/ model
        self.TRAIN_LOGS_SAVE_DIR = f"{self.MODEL_SAVE_DIR}/{model_ts}_history.dat"
        self.VALID_SET_REF_SAVE_DIR = f"{self.MODEL_SAVE_DIR}/{model_ts}_validsetref.dat"
        
    
    def state_dict(self):
        """
        Returns args in this class as a dictionary. Required for
        this class to be saved by the 'accelerate' package along
        with a model.
        """
        return self.__dict__


    def load_state_dict(self, state_dict):
        """
        Loads class args from a dictionary. Required for
        this class to be loaded by the 'accelerate' package, along
        with a model.
        """
        for k, v in state_dict.items():
            setattr(self, k, v)

