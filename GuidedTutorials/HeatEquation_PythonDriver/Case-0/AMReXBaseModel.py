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
        # Create modelpar
        modelpar = self._create_modelpar()
        
        super().__init__(lambda x: x, 
                        ndim=len(modelpar['param_names']) or 1,
                        modelpar=modelpar,
                        **kwargs)
        
        self.field_info = self._create_field_info()
        self.param_names = modelpar['param_names']
        self.output_names = modelpar['output_names']
        print(f"✓ Params: {self.param_names}, Outputs: {self.output_names}")

    def _create_field_info(self):
        """Create yt-style field info container"""
        field_info = {}

        for field_tuple in self._param_fields:
            field_info[field_tuple] = self._get_field_info(field_tuple)

        for field_tuple in self._output_fields:
            field_info[field_tuple] = self._get_field_info(field_tuple)

        return field_info

    def _extract_param_domain(self):
        """Extract parameter bounds into Function-compatible domain array"""
        domain_list = []

        for field_tuple in self._param_fields:
            info = self._get_field_info(field_tuple)
            if 'bounds' in info:
                domain_list.append(info['bounds'])
            else:
                # Use Function's default domain size
                dmax = getattr(self, 'dmax', 10.0)
                domain_list.append([-dmax, dmax])

        if domain_list:
            return np.array(domain_list)
        return None

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

    def _create_modelpar(self):
        """Create basic modelpar dictionary"""
        modelpar = {
            'param_names': [f[1] for f in self._param_fields],
            'output_names': [f[1] for f in self._output_fields],
        }
        return modelpar
