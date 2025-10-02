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

    def forward(self, x):
        """
        PyTorch-compatible forward method for inference.

        Args:
            x: torch.Tensor or np.ndarray
        Returns:
            torch.Tensor if input is tensor, else np.ndarray
        """
        # Check if input is PyTorch tensor
        is_torch = False
        if hasattr(x, 'detach'):  # Duck typing for torch.Tensor
            is_torch = True
            import torch
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        # Run simulation using existing logic
        outputs = self._run_simulation(x_np)

        # Convert back to torch if needed
        if is_torch:
            return torch.from_numpy(outputs).to(x.device)
        return outputs

    def __call__(self, x):
        """Function interface - routes through forward() for consistency"""
        # Ensure x is at least 2D for checkDim
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.checkDim(x)
        if hasattr(self, 'domain') and self.domain is not None:
            self.checkDomain(x)
        
        return self.forward(x)

    def _run_simulation(self, params):
        """
        Core simulation logic - override in subclasses or use evolve/postprocess.
        
        Args:
            params: numpy array of parameters (1D or 2D)
            
        Returns:
            numpy array of outputs
        """
        # Ensure params is 2D (n_samples x n_params)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        
        n_samples = params.shape[0]
        outdim = len(self.output_names) if self.output_names else 1
        outputs = np.zeros((n_samples, outdim))
        
        # Check if subclass has evolve/postprocess methods
        if hasattr(self, 'evolve') and hasattr(self, 'postprocess'):
            for i in range(n_samples):
                # Evolve just returns simulation state
                sim_state = self.evolve(params[i, :])
                # Postprocess extracts outputs from state
                outputs[i, :] = self.postprocess(sim_state)
        else:
            raise NotImplementedError(
                "Must implement _run_simulation or evolve/postprocess methods"
            )
        
        return outputs

    @property
    def field_list(self):
        """All available fields"""
        return list(self.field_info.keys())

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

    def get_param_info(self, param_name):
        """Get info for a specific parameter"""
        for field_tuple in self._param_fields:
            if field_tuple[1] == param_name:
                return self._get_field_info(field_tuple)
        return {}

    def get_output_info(self, output_name):
        """Get info for a specific output"""
        for field_tuple in self._output_fields:
            if field_tuple[1] == output_name:
                return self._get_field_info(field_tuple)
        return {}

    def _create_modelpar(self):
        """Create basic modelpar dictionary"""
        modelpar = {
            'param_names': [f[1] for f in self._param_fields],
            'output_names': [f[1] for f in self._output_fields],
        }
        return modelpar
