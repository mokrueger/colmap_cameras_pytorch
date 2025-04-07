"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import cv2
import numpy as np
from ..models import SimplePinhole


def max_theta_for_image(intrinsics, image_shape):  # TODO: Utils file?
    h, w = image_shape[0], image_shape[1]
    ud, vd = np.meshgrid(np.arange(0, w), np.arange(0, h))
    points = np.stack([ud.ravel(), vd.ravel()], axis=-1)
    normalized = (np.linalg.inv(intrinsics) @ np.concatenate([points, np.ones((points.shape[0], 1))],
                                                             axis=-1).T).T[:, :2]
    r = np.linalg.norm(normalized, axis=-1)
    theta = np.arctan(r)  # the same as np.arctan2(r, np.ones((normalized.shape[0])))
    return theta.max()


class Remapper:

    def __init__(self, step=1):
        self.step = step

    # @staticmethod
    # def determine_monotonicity(model, img_shape):
    #     from sympy import is_monotonic, symbols, Interval, pi, diff  # TODO: requirement
    #     theta = symbols('theta')
    #     f_theta = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
    #
    #     max_theta = max_theta_for_image(model.intrinsics, img_shape)
    #     theta_range = Interval(0, max_theta)
    #     return is_monotonic(f_theta, theta_range)

    def remap(self, model_in, model_out, img, return_intermediates=False, borderValue=None, borderMode=None, interpolation=None):
        w_i, h_i = [int(x.item()) for x in model_out.image_shape]
        device = model_out.device
        assert model_in.device == device, "Models should be on the same device"

        ud, vd = torch.meshgrid(torch.arange(0, w_i, self.step, device=device),
                                torch.arange(0, h_i, self.step, device=device),
                                indexing='xy')
        w = w_i // self.step
        h = h_i // self.step

        points = torch.stack([ud.ravel(), vd.ravel()], dim=-1)
        points3d, valid_out = model_out.unmap(points.float())
        points2d, valid_in = model_in.map(points3d)

        points2d[~valid_in] = -1
        points2d = points2d.reshape(h, w, 2)

        xlut = points2d[..., 0].cpu().numpy().astype(np.float32)  # float32 required by cv2.map
        ylut = points2d[..., 1].cpu().numpy().astype(np.float32)

        if isinstance(img, str):
            img = cv2.imread(img)

        remap_kwargs = {"borderValue": borderValue if borderValue is not None else [np.nan, np.nan, np.nan],
                        "borderMode": borderMode if borderMode is not None else cv2.BORDER_CONSTANT,
                        "interpolation": interpolation if interpolation is not None else cv2.INTER_LINEAR}
        img = cv2.remap(img.astype(np.float32), xlut, ylut, **remap_kwargs)
        img = cv2.resize(img, (w_i, h_i))
        if not return_intermediates:
            return img
        return img, (xlut, ylut), (valid_in & valid_out).cpu().numpy()

    def remap_from_fov(self, model_in, fov_out, img):
        model_out = SimplePinhole.from_fov(fov_out, model_in.image_shape).to(model_in.device)
        return self.remap(model_in, model_out, img)
