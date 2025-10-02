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
        
        # Create field info
        self.field_info = self._create_field_info()
        print(f"✓ Created field_info with {len(self.field_info)} fields")

    def _create_field_info(self):
        """Create yt-style field info container"""
        field_info = {}

        for field_tuple in self._param_fields:
            field_info[field_tuple] = self._get_field_info(field_tuple)

        for field_tuple in self._output_fields:
            field_info[field_tuple] = self._get_field_info(field_tuple)

        return field_info

    def _get_field_info(self, field_tuple):
        """
        Override in subclass to provide field metadata.
        
        Args:
            field_tuple: (field_type, field_name) tuple
            
        Returns:
            dict with 'bounds', 'units', 'mean', 'std', etc.
        """
        # Default empty implementation
        return {}
