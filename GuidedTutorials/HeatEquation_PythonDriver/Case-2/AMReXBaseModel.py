class AMReXModelBase(ModelWrapperFcn):
    # Class-level field definitions
    _field_info_class = None
    _param_fields = []
    _output_fields = []
    _spatial_domain_bounds = None  # Spatial domain bounds (separate from parameter bounds)

    def __init__(self, model=None, **kwargs):
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
        ndim = kwargs.get('ndim', len(self.param_names))
        outdim = kwargs.get('outdim', len(self.output_names))

        # Create model function wrapper
        if model is None:
            model_func = lambda params: self._run_simulation(params)
        else:
            model_func = model

        # Initialize Function with model
        super().__init__(
            model_func,
            ndim,
            modelpar=modelpar,
            name=kwargs.get('name', 'AMReXModel')
        )

        # Set the parameter domain using Function's method
        if param_domain is not None and len(param_domain) > 0:
            self.setDimDom(domain=param_domain)

        # Set output dimension
        self.outdim = outdim

        # Setup spatial domain bounds (yt-style) - separate from parameter bounds
        if self._spatial_domain_bounds:
            self.domain_left_edge = self._spatial_domain_bounds[0]
            self.domain_right_edge = self._spatial_domain_bounds[1]
            self.domain_dimensions = (self._spatial_domain_bounds[2]
                                     if len(self._spatial_domain_bounds) > 2 else None)

        # Initialize AMReX if needed
        self.xp = load_cupy()
        if not amr.initialized():
            amr.initialize([])

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
        self.checkDim(x)
        if hasattr(self, 'domain') and self.domain is not None:
            self.checkDomain(x)
        return self.forward(x)

    def _run_simulation(self, params):
        """Core simulation logic - override in subclasses"""
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
        outputs = np.zeros((n_samples, self.outdim))

        # Default implementation using evolve/postprocess if available
        if hasattr(self, 'evolve') and hasattr(self, 'postprocess'):
            for i in range(n_samples):
                multifab, varnames, geom = self.evolve(params[i, :])
                outputs[i, :] = self.postprocess(multifab, varnames, geom)
        else:
            # Must be overridden in subclass
            raise NotImplementedError("Must implement _run_simulation or evolve/postprocess")

        return outputs

    @property
    def field_list(self):
        """All available fields"""
        return list(self.field_info.keys())

    def _get_field_info(self, field_tuple):
        """Override to provide field metadata"""
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
