import random
import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning import Trainer


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""
        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.add_image('images', images, trainer.global_step)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.add_image('images', images, trainer.global_step)


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