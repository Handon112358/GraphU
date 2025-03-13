RAW_DATA_PATH = 'temp_data/raw_data/'
PROCESSED_DATA_PATH = 'temp_data/processed_data/'
MODEL_PATH = 'temp_data/models/'
ANALYSIS_PATH = 'temp_data/analysis_data/'

# database name
DATABASE_NAME = "unlearning_gnn"

import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False