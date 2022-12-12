import random
import numpy as np
import torch




def set_random_seed(seed):
    """ set random seeds for all possible random libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def state_dict_ckpt(path, device='cpu'):
    if device == 'cpu':
        ckpt = torch.load(path, map_location='cpu')
    else:
        ckpt = torch.load(path)    
    model_state_dict = ckpt["state_dict"]
    dict = model_state_dict.copy()
    for s in dict:
        x = s[6:]
        model_state_dict[x] = model_state_dict.pop(s)
    return model_state_dict