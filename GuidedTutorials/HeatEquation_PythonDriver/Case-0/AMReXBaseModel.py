from pytuq.func.func import ModelWrapperFcn
import numpy as np
from abc import abstractmethod

from pytuq.func.func import ModelWrapperFcn
import numpy as np

class AMReXBaseModel(ModelWrapperFcn):
    """Base class for AMReX models with yt-style field info"""
    
    # Class-level field definitions (to be overridden by subclasses)
    _field_info_class = None
    _param_fields = []
    _output_fields = []
    _spatial_domain_bounds = None

    def __init__(self, **kwargs):
        """Minimal initialization for now"""
        # Just call parent for now
        super().__init__(lambda x: x, ndim=1, **kwargs)
        print("✓ Basic initialization works")

