from pytuq.func.func import ModelWrapperFcn
import numpy as np
from abc import abstractmethod
import os
import tempfile

from pytuq.func.func import ModelWrapperFcn
import numpy as np
import amrex.space3d as amr

def load_cupy():
    """Load appropriate array library (CuPy for GPU, NumPy for CPU)."""
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            amr.Print("Note: found and will use cupy")
            return cp
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Using numpy...")
            import numpy as np
            return np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            return np
    else:
        import numpy as np
        amr.Print("Note: found and will use numpy")
        return np

class AMReXBaseModel(ModelWrapperFcn):
    """Base class for AMReX models with yt-style field info"""

    # Class-level field definitions (to be overridden by subclasses)
    _field_info_class = None
    _param_fields = []
    _output_fields = []
    _spatial_domain_bounds = None

    # Subprocess configuration
    _model_script = './model.x'  # Path to model.x wrapper script
    _use_subprocess = False  # Enable subprocess mode

    def __init__(self, model=None, use_subprocess=False, model_script=None, **kwargs):
        # Initialize AMReX if needed
        self.xp = load_cupy()
        if not amr.initialized():
            amr.initialize([])

        # Configure subprocess mode
        self._use_subprocess = use_subprocess or self.__class__._use_subprocess
        if model_script:
            self._model_script = model_script

        # Create modelpar from existing parameter information
        modelpar = self._create_modelpar()

        # Setup field info container
        self.field_info = self._create_field_info()

        # Setup convenience lists
        self.param_names = [f[1] for f in self._param_fields]
        self.output_names = [f[1] for f in self._output_fields]

        # Extract parameter bounds and create domain array for Function
        param_domain = self._extract_param_domain()

        # Determine dimensions
        ndim = len(self.param_names)
        outdim = len(self.output_names)

        # Create model function wrapper
        if model is None:
            model_func = lambda params, mp=None: self._run_simulation(params)
        else:
            model_func = model

        # Initialize Function with model
        super().__init__(
            model_func,
            ndim,
            modelpar=modelpar,
            name=kwargs.get('name', 'AMReXModel')
        )

        # Set output dimension (ModelWrapperFcn defaults to 1)
        self.outdim = outdim

        # Set the parameter domain using Function's method
        if param_domain is not None and len(param_domain) > 0:
            self.setDimDom(domain=param_domain)

        # Setup spatial domain bounds (yt-style) - separate from parameter bounds
        if self._spatial_domain_bounds:
            self.domain_left_edge = self._spatial_domain_bounds[0]
            self.domain_right_edge = self._spatial_domain_bounds[1]
            self.domain_dimensions = (self._spatial_domain_bounds[2]
                                     if len(self._spatial_domain_bounds) > 2
                                     else None)

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

        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.checkDim(x)
        if hasattr(self, 'domain') and self.domain is not None:
            self.checkDomain(x)

        if self.modelpar is None:
            outputs = self.model(x)
        else:
            outputs = self.model(x, self.modelpar)

        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, self.outdim)  # ← Allows MULTIPLE outputs

        return outputs

    def _run_simulation(self, params):
        """
        Core simulation logic - uses subprocess if enabled.

        Args:
            params: numpy array of parameters (1D or 2D)

        Returns:
            numpy array of outputs
        """
        # Ensure params is 2D (n_samples x n_params)
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
#        outdim = len(self.output_names) if self.output_names else 1

        if self._use_subprocess:
            # Use model.x subprocess approach
            outputs = self._run_subprocess(params)
        else:
            # Original in-process method
            outputs = np.zeros((n_samples, outdim))
            
            if hasattr(self, 'evolve') and hasattr(self, 'postprocess'):
                for i in range(n_samples):
                    sim_state = self.evolve(params[i, :])
                    outputs[i, :] = self.postprocess(sim_state)
            else:
                raise NotImplementedError(
                    "Must implement _run_simulation or evolve/postprocess methods"
                )
        # Ensure we only return outdim outputs
        return outputs[:, :self.outdim]

    def _run_subprocess(self, params):
        """
        Run simulation using model.x subprocess (like online_bb example).

        Args:
            params: 2D numpy array (n_samples x n_params)

        Returns:
            2D numpy array (n_samples x n_outputs)
        """
        # Check if model.x exists
        if not os.path.exists(self._model_script):
            raise FileNotFoundError(f"Model script not found: {self._model_script}")

        # Create temporary input/output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_file = f.name
            np.savetxt(f, params)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_file = f.name

        try:
            # Run model.x
            cmd = f'{self._model_script} {input_file} {output_file}'
            print(f"Running: {cmd}")
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                raise RuntimeError(f"Command failed with exit code {exit_code}: {cmd}")

            # Load outputs
            outputs = np.loadtxt(output_file).reshape(params.shape[0], -1)
            
            return outputs

        finally:
            # Clean up temporary files
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

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
        """Create modelpar dictionary with statistical properties"""
        modelpar = {
            'param_info': {},
            'output_info': {},
            'param_names': [],
            'output_names': [],
            'defaults': {},
            'bounds': {},
            'units': {},
            'mean': [],
            'std': [],
            'distribution': [],  # 'normal', 'uniform', 'lognormal', etc.
            'pc_type': 'HG',  # Hermite-Gaussian by default
        }

        # Extract parameter information including statistical properties
        for field_tuple in self._param_fields:
            field_type, field_name = field_tuple
            info = self._get_field_info(field_tuple)

            modelpar['param_info'][field_name] = info
            modelpar['param_names'].append(field_name)

            # Extract statistical properties
            if 'mean' in info:
                modelpar['mean'].append(info['mean'])
            elif 'default' in info:
                modelpar['mean'].append(info['default'])
            else:
                # Use center of bounds if available
                if 'bounds' in info:
                    modelpar['mean'].append(np.mean(info['bounds']))
                else:
                    modelpar['mean'].append(0.0)

            if 'std' in info:
                modelpar['std'].append(info['std'])
            else:
                # Default to 10% of mean or range
                if 'bounds' in info:
                    # Use range/6 as rough std (99.7% within bounds)
                    modelpar['std'].append((info['bounds'][1] - info['bounds'][0])/6.0)
                else:
                    modelpar['std'].append(abs(modelpar['mean'][-1]) * 0.1)

            # Store other properties
            if 'bounds' in info:
                modelpar['bounds'][field_name] = info['bounds']
            if 'units' in info:
                modelpar['units'][field_name] = info['units']
            if 'distribution' in info:
                modelpar['distribution'].append(info['distribution'])
            else:
                modelpar['distribution'].append('normal')  # default

        # Convert to numpy arrays for easier manipulation
        modelpar['mean'] = np.array(modelpar['mean'])
        modelpar['std'] = np.array(modelpar['std'])

        # Add output information
        for field_tuple in self._output_fields:
            field_type, field_name = field_tuple
            info = self._get_field_info(field_tuple)
            modelpar['output_info'][field_name] = info
            modelpar['output_names'].append(field_name)

        return modelpar

    def write_param_marginals(self, filename='param_margpc.txt'):
        """Write parameter marginals file for PC analysis"""
        with open(filename, 'w') as f:
            for i, name in enumerate(self.modelpar['param_names']):
                mean = self.modelpar['mean'][i]
                std = self.modelpar['std'][i]
                f.write(f"{mean} {std}\n")
        print(f"Wrote parameter marginals to {filename}")
