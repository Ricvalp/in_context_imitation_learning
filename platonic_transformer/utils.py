import math
import numbers
import gc
import os.path as osp
import random
import time
from typing import Tuple, Union, Optional,Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, LinearTransformation

def run_gc():
    gc.collect()
    torch.cuda.empty_cache()




def format_batch_for_esen(batch):
    """
    Convert an OMol batch to the format expected by the eSEN model.
    
    Args:
        batch: Batch object from OMol dataloader
        
    Returns:
        AtomicDataObj: Object with keys expected by eSEN model
    """
    
    # Create a class that supports both dict and attribute access
    class AtomicDataObj:
        def __init__(self, **kwargs):
            # Store data as attributes
            for key, value in kwargs.items():
                setattr(self, key, value)
            # Store internal dictionary for dict-like access
            self._dict = kwargs
        
        def __getitem__(self, key):
            # Support dictionary-style access
            if key in self._dict:
                return self._dict[key]
            elif hasattr(self, str(key)):  # Convert key to string for attribute access
                return getattr(self, str(key))
            raise KeyError(f"Key '{key}' not found")
        
        def __setitem__(self, key, value):
            # Support dictionary-style assignment
            self._dict[key] = value
            # Only set as attribute if key is string or can be converted to valid attribute name
            if isinstance(key, str):
                setattr(self, key, value)
            elif isinstance(key, int):
                # For integer keys, store in dict only (don't create attributes)
                pass
            else:
                # For other types, try to convert to string
                try:
                    setattr(self, str(key), value)
                except (TypeError, ValueError):
                    pass  # If conversion fails, just store in dict
        
        def get(self, key, default=None):
            # Support .get() method like dictionaries
            return self._dict.get(key, default)
        
        def keys(self):
            # Support .keys() method
            return self._dict.keys()
        
        def __contains__(self, key):
            # Support 'in' operator
            return key in self._dict
        
        def __len__(self):
            # Return the number of systems in the batch
            if hasattr(self, 'natoms'):
                return len(self.natoms)
            return 0  # Fallback if natoms is not available
    
    # Create systems list needed by eSEN
    systems = []
    natoms = batch.num_atoms.tolist()
    
    start_idx = 0
    for i, num_atoms in enumerate(natoms):
        end_idx = start_idx + num_atoms
        
        # Extract data for this system as an AtomicDataObj (not dict)
        system = AtomicDataObj(
            pos=batch.pos[start_idx:end_idx],
            atomic_numbers=batch.atomic_numbers[start_idx:end_idx],
            cell=batch.cell[i:i+1] if hasattr(batch, 'cell') else torch.eye(3, device=batch.pos.device).unsqueeze(0) * 20.0,
            pbc=torch.zeros(1, 3, dtype=torch.bool, device=batch.pos.device),
            natoms=torch.tensor([num_atoms], device=batch.pos.device)
        )
        systems.append(system)
        start_idx = end_idx
    
    # Prepare main data object for eSEN
    data_dict = AtomicDataObj(
        pos=batch.pos,
        atomic_numbers=batch.atomic_numbers,
        batch=batch.batch,
        natoms=torch.tensor(natoms, device=batch.pos.device),
        charge=torch.zeros(len(natoms), device=batch.pos.device) if not hasattr(batch, 'charge') else batch.charge,
        spin=torch.zeros(len(natoms), device=batch.pos.device) if not hasattr(batch, 'spin') else batch.spin,
        dataset=["default"] * len(natoms),
        systems=systems
    )
    
    # Add cell and pbc if available
    if hasattr(batch, 'cell'):
        data_dict.cell = batch.cell
    else:
        data_dict.cell = torch.eye(3, device=batch.pos.device).unsqueeze(0).repeat(len(natoms), 1, 1) * 20.0
        
    if hasattr(batch, 'pbc'):
        data_dict.pbc = batch.pbc
    else:
        data_dict.pbc = torch.zeros(len(natoms), 3, dtype=torch.bool, device=batch.pos.device)
    
    # Add individual systems by numerical index (important for graph generation)
    for i, system in enumerate(systems):
        data_dict[i] = system  # This will only store in _dict, not as attribute

    return data_dict



class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) * 1.0 / (self.warmup + 1e-6)
        return lr_factor



class RandomRotateWithNormals(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    def __init__(self, degrees: Union[Tuple[float, float], float],
                 axis: int = 0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data: Data) -> Data:
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformationWithNormals(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')
    
class LinearTransformationWithNormals(BaseTransform):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :obj:`[D, D]` where :obj:`D`
            corresponds to the dimensionality of node positions.
    """
    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        # Store the matrix as its transpose.
        # We do this to enable post-multiplication in `__call__`.
        self.matrix = matrix.t()

    def __call__(self, data):
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
        norm = data.x.view(-1, 1) if data.x.dim() == 1 else data.x

        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        assert norm.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        # We post-multiply the points by the transformation matrix instead of
        # pre-multiplying, because `data.pos` has shape `[N, D]`, and we want
        # to preserve this shape.
        data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))
        data.x = torch.matmul(norm, self.matrix.to(norm.dtype).to(norm.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())

# Adapted from pytorch geometric, but updated the cross product as not to procude warnings
class SamplePoints(BaseTransform):
    """Uniformly samples a fixed number of points on the mesh faces according
    to their face area.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """
    def __init__(
        self,
        num: int,
        remove_faces: bool = True,
        include_normals: bool = False,
    ):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None

        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        # Normalize positions
        pos_max = pos.abs().max()
        pos = pos / pos_max

        # Calculate face areas using linalg.cross
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        area = torch.linalg.cross(vec1, vec2, dim=1)
        area = area.norm(p=2, dim=1).abs() / 2

        # Sample points based on face areas
        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        # Generate random barycentric coordinates
        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        # Calculate vectors for point sampling
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        # Compute normals if requested
        if self.include_normals:
            normals = torch.linalg.cross(vec1, vec2, dim=1)
            data.normal = torch.nn.functional.normalize(normals, p=2, dim=1)

        # Sample points using barycentric coordinates
        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        # Restore original scale
        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data
    
class NormalizeCoord(BaseTransform):
    """
    Normalizes the point cloud coordinates by:
    1. Centering them at the origin (zero mean)
    2. Scaling by the maximum distance from origin
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # Center the points by subtracting mean
        centroid = torch.mean(data.pos, dim=0)
        data.pos = data.pos - centroid

        # Scale by maximum distance from origin
        distances = torch.sqrt(torch.sum(data.pos ** 2, dim=1))
        scale = torch.max(distances)
        data.pos = data.pos / scale

        return data

class RandomJitter(BaseTransform):
    """Randomly jitter points by adding normal noise."""
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        noise = torch.clamp(
            self.sigma * torch.randn_like(data.pos), 
            min=-self.clip, 
            max=self.clip
        )
        data.pos = data.pos + noise
        return data

class RandomShift(BaseTransform):
    """Randomly shift the point cloud."""
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = torch.rand(3) * 2 * self.shift_range - self.shift_range
        shift = shift.to(data.pos.device)
        data.pos = data.pos + shift
        return data
    
class RandomRotatePerturbation(BaseTransform):
    """Apply small random rotations around all axes."""
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data):
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3),
            min=-self.angle_clip,
            max=self.angle_clip
        ).to(data.pos.device)

        # Create rotation matrices for each axis
        cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

        Rx = torch.tensor([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]], device=data.pos.device)

        Ry = torch.tensor([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]], device=data.pos.device)

        Rz = torch.tensor([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]], device=data.pos.device)

        R = torch.mm(torch.mm(Rz, Ry), Rx)
        
        # Apply rotation
        data.pos = torch.mm(data.pos, R.t())
        if hasattr(data, 'normal'):
            data.normal = torch.mm(data.normal, R.t())
            
        return data
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def scatter_mean(src, index, dim, dim_size):
    # Step 1: Perform scatter add (sum)
    out_shape = [dim_size] + list(src.shape[1:])
    out_sum = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    out_sum.scatter_add_(dim, index_expanded, src)
    
    # Step 2: Count occurrences of each index to calculate the mean
    ones = torch.ones_like(src)
    out_count = torch.zeros(out_shape, dtype=torch.float, device=src.device)
    out_count.scatter_add_(dim, index_expanded, ones)
    out_count[out_count == 0] = 1  # Avoid division by zero
    
    # Calculate mean by dividing sum by count
    out_mean = out_sum / out_count

    return out_mean

def fully_connected_edge_index(batch_idx):
    edge_indices = []
    for batch_num in torch.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing='ij')
        edge_indices.append(torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0))
    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index

def subtract_mean(pos, batch):
    means = scatter_mean(src=pos, index=batch, dim=0, dim_size=batch.max().item()+1)
    return pos - means[batch]


class RandomSOd(torch.nn.Module):
        def __init__(self, d):
            """
            Initializes the RandomRotationGenerator.
            Args:
            - d (int): The dimension of the rotation matrices (2 or 3).
            """
            super(RandomSOd, self).__init__()
            assert d in [2, 3], "d must be 2 or 3."
            self.d = d

        def forward(self, n=None):
            """
            Generates random rotation matrices.
            Args:
            - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.
            
            Returns:
            - Tensor: A tensor of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
            """
            if self.d == 2:
                return self._generate_2d(n)
            else:
                return self._generate_3d(n)
        
        def _generate_2d(self, n):
            theta = torch.rand(n) * 2 * torch.pi if n else torch.rand(1) * 2 * torch.pi
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
            if n:
                return rotation_matrix.view(n, 2, 2)
            return rotation_matrix.view(2, 2)

        def _generate_3d(self, n):
            q = torch.randn(n, 4) if n else torch.randn(4)
            q = q / torch.norm(q, dim=-1, keepdim=True)
            q0, q1, q2, q3 = q.unbind(-1)
            rotation_matrix = torch.stack([
                1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
                2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1),
                2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)
            ], dim=-1)
            if n:
                return rotation_matrix.view(n, 3, 3)
            return rotation_matrix.view(3, 3)

class RandomSO2AroundAxis(torch.nn.Module):
    def __init__(self, axis=2, degrees=15):
        """
        Initializes a generator for rotations around a specific axis.
        Args:
        - axis (int): The rotation axis (0=X, 1=Y, 2=Z)
        - degrees (float or tuple): Maximum rotation angle in degrees.
                                  If float, uses (-|degrees|, |degrees|).
                                  If tuple, uses (min_degrees, max_degrees).
        """
        super().__init__()
        assert axis in [0, 1, 2], "axis must be 0 (X), 1 (Y), or 2 (Z)"
        self.axis = axis
        
        # Handle degrees argument
        if isinstance(degrees, (float, int)):
            self.degrees = (-abs(float(degrees)), abs(float(degrees)))
        elif isinstance(degrees, (tuple, list)):
            assert len(degrees) == 2, "degrees tuple must have length 2"
            self.degrees = tuple(map(float, degrees))
        else:
            raise ValueError("degrees must be a number or a tuple")

    def forward(self, n=None):
        """
        Generates random rotation matrices around the specified axis.
        Args:
        - n (int, optional): The number of rotation matrices to generate. 
                            If None, generates a single matrix.
        
        Returns:
        - Tensor: A tensor of shape [n, 3, 3] containing n rotation matrices, 
                 or [3, 3] if n is None.
        """
        # Generate random angles in degrees
        min_deg, max_deg = self.degrees
        angles = torch.rand(n if n else 1) * (max_deg - min_deg) + min_deg
        # Convert to radians
        theta = angles * torch.pi / 180.0
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        
        # Create rotation matrices based on axis
        if self.axis == 0:  # X-axis
            rotation_matrix = torch.stack([
                torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), cos_theta, sin_theta,
                torch.zeros_like(cos_theta), -sin_theta, cos_theta
            ], dim=-1)
        elif self.axis == 1:  # Y-axis
            rotation_matrix = torch.stack([
                cos_theta, torch.zeros_like(cos_theta), -sin_theta,
                torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta),
                sin_theta, torch.zeros_like(cos_theta), cos_theta
            ], dim=-1)
        else:  # Z-axis
            rotation_matrix = torch.stack([
                cos_theta, sin_theta, torch.zeros_like(cos_theta),
                -sin_theta, cos_theta, torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)
            ], dim=-1)
        
        if n:
            return rotation_matrix.view(n, 3, 3)
        return rotation_matrix.view(3, 3)

class TimerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.total_training_start_time = 0.0
        self.epoch_start_time = 0.0
        self.test_inference_time = 0.0

    # Called when training begins
    def on_train_start(self, trainer, pl_module):
        self.total_training_start_time = time.time()

    # Called when training ends
    def on_train_end(self, trainer, pl_module):
        total_training_time = (time.time() - self.total_training_start_time)/60
        # Log total training time at the end of training
        trainer.logger.experiment.log({"Total Training Time (min)" : total_training_time})

    # Called at the start of the test epoch
    def on_test_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    # Called at the end of the test epoch
    def on_test_epoch_end(self, trainer, pl_module):
        # Calculate the inference time for the entire test epoch
        self.test_inference_time = (time.time() - self.epoch_start_time)/60
        # Log the inference time for the test epoch
        trainer.logger.experiment.log({"Test Inference Time (min)": self.test_inference_time})

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) / (self.warmup + 1e-6)
        return lr_factor


class StopOnPersistentDivergence(pl.Callback):
    """
    Callback to stop training if a monitored metric stays above a specific
    divergence threshold for a given number of consecutive validation epochs,
    after an initial grace period.

    Args:
        monitor (str): Quantity to be monitored (e.g., "val MAE").
        threshold (float): The divergence threshold. Training stops if the metric
                           is strictly greater than this value persistently.
        patience (int): Number of consecutive validation epochs the metric must be
                        above the threshold (after the grace period) before
                        training is stopped.
        grace_epochs (int): Number of initial training epochs during which this
                            divergence check is completely ignored.
        verbose (bool): If True, prints messages when the callback takes action
                        or is in a specific state.
    """
    def __init__(self,
                 monitor: str = "val MAE",
                 threshold: float = 0.8,
                 patience: int = 3,
                 grace_epochs: int = 10,
                 verbose: bool = False):
        super().__init__()

        if not isinstance(monitor, str) or not monitor:
            raise ValueError("Argument `monitor` must be a non-empty string.")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Argument `threshold` must be a number.")
        if not isinstance(patience, int) or patience < 1: # Patience must be at least 1
            raise ValueError("Argument `patience` must be an integer greater than or equal to 1.")
        if not isinstance(grace_epochs, int) or grace_epochs < 0:
            raise ValueError("Argument `grace_epochs` must be a non-negative integer.")

        self.monitor = monitor
        self.threshold = threshold
        self.patience = patience
        self.grace_epochs = grace_epochs
        self.verbose = verbose

        self.consecutive_exceeds_count = 0
        self.stopped_epoch = 0 # To record when stopping occurs

    def _check_metric(self, trainer: "pl.Trainer") -> Optional[float]:
        """Helper to get the metric value from trainer.callback_metrics or trainer.logged_metrics."""
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            metrics = trainer.logged_metrics # Fallback to logged_metrics

        if self.monitor in metrics:
            metric_val = metrics[self.monitor]
            # Ensure metric_val is a scalar float
            if hasattr(metric_val, 'item'): # For tensors
                return metric_val.item()
            try:
                return float(metric_val)
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"{self.__class__.__name__}: Warning: Metric '{self.monitor}' has non-convertible type {type(metric_val)}.")
                return None
        return None

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        current_epoch = trainer.current_epoch

        if current_epoch < self.grace_epochs:
            if self.verbose and current_epoch == 0: # Print only once at the start of training
                print(f"{self.__class__.__name__}: In grace period (first {self.grace_epochs} epochs). "
                      f"Divergence check for '{self.monitor}' (threshold > {self.threshold}) is inactive.")
            return # Skip the check during the grace period

        current_metric_value = self._check_metric(trainer)

        if current_metric_value is None:
            if self.verbose and current_epoch == self.grace_epochs: # Print warning once when check becomes active
                print(f"{self.__class__.__name__}: Metric '{self.monitor}' not found in logs at epoch {current_epoch} "
                      f"(when divergence check became active). "
                      f"Available callback_metrics: {list(trainer.callback_metrics.keys())}. "
                      f"Available logged_metrics: {list(trainer.logged_metrics.keys())}.")
            return

        status_message = (
            f"{self.__class__.__name__} at epoch {current_epoch}: '{self.monitor}' = {current_metric_value:.4f}. "
            f"Threshold: > {self.threshold}. Consecutive exceeds: {self.consecutive_exceeds_count}. Patience: {self.patience}."
        )

        if current_metric_value > self.threshold:
            self.consecutive_exceeds_count += 1
            if self.verbose:
                print(f"{status_message} Metric EXCEEDED threshold.")
        else:
            if self.verbose and self.consecutive_exceeds_count > 0: # Log reset only if it was counting
                print(f"{status_message} Metric NOT ABOVE threshold. Resetting consecutive count.")
            self.consecutive_exceeds_count = 0 # Reset count

        if self.consecutive_exceeds_count >= self.patience:
            self.stopped_epoch = current_epoch
            trainer.should_stop = True # Signal trainer to stop
            if self.verbose:
                print(f"\n{self.__class__.__name__}: Stopping training at epoch {self.stopped_epoch} "
                      f"because '{self.monitor}' ({current_metric_value:.4f}) "
                      f"was above divergence threshold ({self.threshold}) "
                      f"for {self.consecutive_exceeds_count} consecutive epochs (patience={self.patience}) "
                      f"after grace period of {self.grace_epochs} epochs.")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.stopped_epoch > 0 and self.verbose:
            print(f"{self.__class__.__name__}: Training ended. Stopped early at epoch {self.stopped_epoch} "
                  f"due to persistent divergence of '{self.monitor}'.")


class TrainingTimerCallback(pl.Callback):
    """
    Callback to measure:
    1. Forward function time during validation (gradients off)
    2. Full training cycle time during training (forward + backward + optimizer)
    """
    
    def __init__(self, num_epochs_to_measure: int = 3, forward_function: str = "forward", units: str = "ms"):
        super().__init__()
        self.num_epochs_to_measure = num_epochs_to_measure
        self.forward_function = forward_function
        self.units = units
        self.multiplier = 1000 if units == "ms" else 1
        
        # Storage for timing data
        self.validation_forward_times = []  # Forward function time during validation
        self.training_full_times = []       # Full training time during training
        
        self.measuring = False
        self.epochs_measured = 0
        
        # Per-epoch storage
        self.current_val_forward_times = []
        self.current_train_full_times = []
        
        # Function wrapping
        self.original_forward_function = None
        self.batch_start_time = None
        
        # Flags to track which phase we're in
        self.in_validation = False
        self.in_training = False
    
    def _should_measure(self):
        """Check if we should still be measuring."""
        return self.epochs_measured < self.num_epochs_to_measure
    
    def _wrap_forward_function(self, pl_module: pl.LightningModule):
        """Wrap the forward function to measure its execution time during validation."""
        if not hasattr(pl_module, self.forward_function):
            print(f"Warning: {self.forward_function} method not found in module")
            return
            
        if hasattr(self, 'function_wrapped') and self.function_wrapped:
            return

        self.original_forward_function = getattr(pl_module, self.forward_function)
        self.function_wrapped = True
        
        def timed_forward(*args, **kwargs):
            if self.measuring and self.in_validation:
                start_time = time.perf_counter()
                result = self.original_forward_function(*args, **kwargs)
                end_time = time.perf_counter()
                forward_time = (end_time - start_time) * self.multiplier
                self.current_val_forward_times.append(forward_time)
                return result
            else:
                return self.original_forward_function(*args, **kwargs)
        
        setattr(pl_module, self.forward_function, timed_forward)
    
    def _restore_forward_function(self, pl_module: pl.LightningModule):
        """Restore the original forward function."""
        if self.original_forward_function is not None:
            setattr(pl_module, self.forward_function, self.original_forward_function)
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._should_measure():
            self.current_train_full_times = []
            self.measuring = True
            self.in_training = True
            self._wrap_forward_function(pl_module)
    
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int):
        if self.measuring and self.in_training:   
            self.batch_start_time = time.perf_counter()
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self.measuring and self.in_training and self.batch_start_time is not None:
            batch_end_time = time.perf_counter()
            full_training_time = (batch_end_time - self.batch_start_time) * self.multiplier
            self.current_train_full_times.append(full_training_time)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.measuring and self.in_training:
            self.in_training = False
            if self.current_train_full_times:
                self.training_full_times.extend(self.current_train_full_times)
    
   
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._should_measure():
            self.current_val_forward_times = []
            self.in_validation = True
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._should_measure() and self.in_validation:
            self.in_validation = False
            if self.current_val_forward_times:
                self.validation_forward_times.extend(self.current_val_forward_times)
            self.epochs_measured += 1
            if self.epochs_measured >= self.num_epochs_to_measure:
                self._log_final_results(pl_module)
                self.measuring = False
                self._restore_forward_function(pl_module)
    
    def _log_final_results(self, pl_module: pl.LightningModule):
        """Log final average timings across all measured epochs."""
        if self.training_full_times:
            avg_training_time = np.mean(self.training_full_times)
            std_training_time = np.std(self.training_full_times)
            
            pl_module.log(f"avg_forward_backward_optimizer_time_[{self.units}]", avg_training_time, 
                         prog_bar=True, logger=True)

            print(f"Average (forward+backward+optimizer) time over {self.epochs_measured} epochs: "
                  f"{avg_training_time:.2f} ± {std_training_time:.2f} {self.units}")
        
        if self.validation_forward_times:
            avg_validation_time = np.mean(self.validation_forward_times)
            std_validation_time = np.std(self.validation_forward_times)
            
            pl_module.log(f"avg_{self.forward_function}_time_[{self.units}]", avg_validation_time, 
                         prog_bar=True, logger=True)
            
            print(f"Average {self.forward_function} time over {self.epochs_measured} epochs: "
                  f"{avg_validation_time:.2f} ± {std_validation_time:.2f} {self.units}")

class MemoryMonitorCallback(pl.Callback):
    def __init__(self, log_frequency=500):
        super().__init__()
        self.log_frequency = log_frequency
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_frequency == 0:
            # Log system memory
            import psutil
            process = psutil.Process()
            ram_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            pl_module.log("system_memory_gb", ram_gb, prog_bar=False)
            
            # Log GPU memory if available
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
                pl_module.log("gpu_memory_gb", gpu_gb, prog_bar=False)

class NaNDetectorCallback(pl.callbacks.Callback):
    def __init__(self, check_gradients=True, check_parameters=True, check_loss=True, 
                 terminate_on_nan=True, log_module_outputs=True):
        super().__init__()
        self.check_gradients = check_gradients
        self.check_parameters = check_parameters
        self.check_loss = check_loss
        self.terminate_on_nan = terminate_on_nan
        self.log_module_outputs = log_module_outputs
        self.module_outputs = {}
        self.nan_detected = False
        
    def _register_hooks(self, pl_module):
        if not self.log_module_outputs:
            return
            
        def hook_fn(module, input, output, name):
            # Check if output contains NaN
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any()
            elif isinstance(output, tuple) or isinstance(output, list):
                has_nan = any(torch.isnan(x).any() for x in output if isinstance(x, torch.Tensor))
            else:
                has_nan = False
                
            if has_nan:
                self.nan_detected = True
                message = f"NaN detected in output of module: {name}"
                print(f"\n{'='*80}\n{message}\n{'='*80}\n")
                
                if self.terminate_on_nan:
                    raise ValueError(message)
        
        # Register hooks for all modules
        self.hooks = []
        for name, module in pl_module.named_modules():
            if name:  # Skip the root module
                hook = module.register_forward_hook(
                    lambda mod, inp, outp, name=name: hook_fn(mod, inp, outp, name)
                )
                self.hooks.append(hook)
    
    def on_fit_start(self, trainer, pl_module):
        self._register_hooks(pl_module)
        
    def _check_tensor_for_nan(self, tensor, tensor_name, pl_module=None):
        if tensor is None:
            return False
            
        if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
            self.nan_detected = True
            message = f"NaN detected in {tensor_name}"
            print(f"\n{'='*80}\n{message}\n{'='*80}\n")
            
            if self.terminate_on_nan:
                raise ValueError(message)
            return True
        return False
    
    def on_before_backward(self, trainer, pl_module, loss):
        if self.check_loss:
            self._check_tensor_for_nan(loss, 'loss')
    
    def on_after_backward(self, trainer, pl_module):
        if self.check_gradients:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self._check_tensor_for_nan(param.grad, f"gradient of {name}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.check_parameters:
            for name, param in pl_module.named_parameters():
                self._check_tensor_for_nan(param.data, f"parameter {name}")
    
    def on_fit_end(self, trainer, pl_module):
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()

