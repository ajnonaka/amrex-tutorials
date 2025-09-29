from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional, Union
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


class SimulationModel(ABC):
    """Base class defining the simulation interface."""

    def __init__(self, **kwargs):
        """Initialize AMReX if needed."""
        if not amr.initialized():
            amr.initialize([])

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Run simulation for each parameter set.

        Parameters:
        -----------
        params : np.ndarray of shape (n_samples, n_params) or (n_params,)
            Parameter sets to run

        Returns:
        --------
        np.ndarray of shape (n_samples, n_outputs)
            Simulation outputs
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
        n_outputs = len(self.get_outnames())
        outputs = np.zeros((n_samples, n_outputs))

        for i in range(n_samples):
            try:
                multifab, varnames, geom = self.evolve(params[i, :])
                outputs[i, :] = self.postprocess(multifab, varnames, geom)
            except Exception as e:
                amr.Print(f"Warning: Simulation failed for parameter set {i}: {e}")
                outputs[i, :] = np.nan

        return outputs

    @abstractmethod
    def evolve(self, param_set: np.ndarray) -> Tuple[amr.MultiFab, Optional[amr.Vector_string], Optional[amr.Geometry]]:
        """
        Run a single simulation step/evolution.
        
        Parameters:
        -----------
        param_set : np.ndarray
            Single parameter set
            
        Returns:
        --------
        tuple : (multifab, varnames, geom)
            - multifab: simulation data
            - varnames: variable names or None
            - geom: domain geometry or None
        """
        pass

    def postprocess(self, multifab: amr.MultiFab, varnames: Optional[amr.Vector_string], 
                   geom: Optional[amr.Geometry]) -> np.ndarray:
        """
        Basic postprocessing with simple one-line operations.
        
        Parameters:
        -----------
        multifab : amr.MultiFab
            Simulation data
        varnames : amr.Vector_string or None
            Variable names (unused in base implementation)
        geom : amr.Geometry or None
            Domain geometry (unused in base implementation)
            
        Returns:
        --------
        np.ndarray
            Processed outputs [max, sum, l2_norm]
        """
        max_val = multifab.max(comp=0, local=False)
        sum_val = multifab.sum(comp=0, local=False)
        l2_norm = multifab.norm2(0)
        
        return np.array([max_val, sum_val, l2_norm])

    @abstractmethod
    def get_pnames(self) -> List[str]:
        """Get parameter names."""
        pass

    @abstractmethod
    def get_outnames(self) -> List[str]:
        """Get output names."""
        pass
