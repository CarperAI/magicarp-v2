from magicarp.losses import _LOSSES

from magicarp.losses.ranking import *
from magicarp.losses.pairwise import *

def get_loss(name):
    """Get a loss function by name
    Args:
        name: Name of the loss
    """
    return _LOSSES[name.lower()]

