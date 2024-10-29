"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch

class BaseModel:
    """
    Partially copilot generated class that behaves almost like a torch.Tensor
    """
    _image_shape : torch.Tensor
    _data : torch.Tensor
    ROOT_FINDING_MAX_ITERATIONS = 100
    OPTIMIZATION_FIX_FOCALS = False
    OPTIMIZATION_FIX_CENTER = True
    OPTIMIZATION_FIX_EXTRA = False
    EPSILON = 1e-6

    def __init__(self, data, image_shape):
        self._data = data
        self._image_shape = image_shape
        if self.num_extra_params == -1:
            if data.shape[0] < self.num_focal_params + self.num_pp_params + 1:
                raise ValueError(f"Expected at least {self.num_focal_params + self.num_pp_params + 1} parameters, got {data.shape[0]}")
            self.num_extra_params = data.shape[0] - self.num_focal_params - self.num_pp_params

        if data.shape[0] != self.num_focal_params + self.num_pp_params + self.num_extra_params:
            raise ValueError(f"Expected {self.num_focal_params + self.num_pp_params + self.num_extra_params} parameters, got {data.shape[0]}")
    
    def __repr__(self):
        f = f'{self.model_name}' + ': {'
        image_size = f'{self._image_shape.tolist()}'
        focals = f'{self[:self.num_focal_params].tolist()}'
        pp = f'{self[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()}'
        extra = f'{self[self.num_focal_params + self.num_pp_params:].tolist()}'
        f = f + '\n\timage_size: ' + image_size
        f = f + '\n\tfocals: ' + focals
        f = f + '\n\tpp: ' + pp
        f = f + '\n\textra: ' + extra + '\n}'
        return f

    def map(self, points3d):
        raise NotImplementedError

    def unmap(self, points2d):
        raise NotImplementedError

    @property
    def image_shape(self):
        return self._image_shape
    
    def check_bounds(self, points2d):
        return (points2d >= 0).all(dim=-1) & (points2d[:, 0] <= self._image_shape[0]-1) & (points2d[:, 1] <= self._image_shape[1]-1)   
    def clone(self, *args, **kwargs):
        return type(self)(self._data.clone(*args, **kwargs), self._image_shape.clone(*args, **kwargs))
    
    def __getitem__(self, idx):
        focal_end = self.num_focal_params
        pp_end = focal_end + self.num_pp_params
        if self.OPTIMIZATION_FIX_CENTER or self.OPTIMIZATION_FIX_FOCALS or self.OPTIMIZATION_FIX_EXTRA:
            if type(idx) == slice:
                if idx.start == None: idx = slice(0, idx.stop)
                if idx.stop == None: idx = slice(idx.start, self._data.shape[0])
                
                if self.OPTIMIZATION_FIX_FOCALS and idx.start < focal_end and idx.stop <= focal_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_CENTER and idx.start >= focal_end and idx.stop <= pp_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_EXTRA and idx.start >= pp_end and idx.stop <= self._data.shape[0]:
                    return self._data[idx].detach()

            else:
                if self.OPTIMIZATION_FIX_FOCALS and idx < focal_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_CENTER and idx >= focal_end and idx < pp_end:
                    return self._data[idx].detach()
                if self.OPTIMIZATION_FIX_EXTRA and idx >= pp_end and idx < self._data.shape[0]:
                    return self._data[idx].detach()

        return  self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    def cpu(self):
        return type(self)(self._data.cpu(), self._image_shape.cpu())
    
    def cuda(self, device=None):
        return type(self)(self._data.cuda(device), self._image_shape.cuda(device))

    def detach(self):
        return type(self)(self._data.detach(), self._image_shape.detach())

    def to(self, *args, **kwargs):
        return type(self)(self._data.to(*args, **kwargs), self._image_shape.to(*args, **kwargs))
    
    def to_colmap(self):
        sh = list(map(str, self._image_shape.tolist()))
        fp = list(map(str, self[:self.num_focal_params].tolist()))
        pp = list(map(str, self[self.num_focal_params:self.num_focal_params + self.num_pp_params].tolist()))
        ep = list(map(str, self[self.num_focal_params + self.num_pp_params:].tolist()))
        f = f'{self.model_name} ' + ' '.join(sh) + ' '
        f = f + ' '.join(fp) + ' ' + ' '.join(pp) + ' ' + ' '.join(ep)
        return f
    
    def get_focal(self):
        return self[:self.num_focal_params].mean()
    def get_center(self):
        return self[self.num_focal_params:self.num_focal_params + self.num_pp_params]

    @property
    def dtype(self): return self._data.dtype
    @property
    def device(self): return self._data.device
    @property
    def ndim(self): return self._data.ndim
    @property
    def shape(self): return self._data.shape
    @property
    def size(self): return self._data.size()

    def get_requires_grad(self):
        return self._data.requires_grad
    def set_requires_grad(self, value: bool):
        self._data.requires_grad = value
    requires_grad = property(get_requires_grad, set_requires_grad)
