"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
from ..base_model import BaseModel
import torch
import numpy as np
import math
from ..utils.newton_root_1d import NewtonRoot1D


def normalize_points2d(points2d, intrinsics):
    inv_K = np.linalg.inv(intrinsics)
    return (inv_K @ np.concatenate([points2d, np.ones((points2d.shape[0], 1))], axis=-1).T).T[:, :2]


def max_theta_for_map_from_points(points2d, intrinsics, h, w):
    normalized = normalize_points2d(points2d, intrinsics)
    r = np.linalg.norm(normalized, axis=-1)
    theta = np.arctan(r)  # the same as np.arctan2(r, np.ones((normalized.shape[0])))
    return theta.max().item()


def is_monotonous_polynomial(distortion, max_theta=None):
    """ This function checks if the distortion function is monotonous up to max_theta (default=pi/2) """
    # from sympy import is_monotonic, symbols, Interval, pi, diff
    # theta = symbols('theta')
    # k1, k2, k3, k4 = distortion
    # f_theta = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9
    #
    # if max_theta is None:
    #     max_theta = pi/2
    # theta_range = Interval(0, max_theta)
    # return is_monotonic(f_theta, theta_range)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # this is the derivative of the fisheye distortion stuff
    # 1 + (3 * k1 * θ^2) + (5 * k1 * θ^4) + (7 * k1 * θ^6) + (9 * k1 * θ^8)
    polynomial = torch.zeros(9).to(device)
    polynomial[0] = torch.tensor(distortion[3]).to(device) * 9  # x^8
    polynomial[1] = 0  # x^7
    polynomial[2] = torch.tensor(distortion[2]).to(device) * 7  # x^6
    polynomial[3] = 0  # x^5
    polynomial[4] = torch.tensor(distortion[1]).to(device) * 5  # x^4
    polynomial[5] = 0  # x^3
    polynomial[6] = torch.tensor(distortion[0]).to(device) * 3  # x^2
    polynomial[7] = 0  # x^1
    polynomial[8] = 1  # x^0

    degree = polynomial.size(0) - 1

    companion_matrix = torch.zeros(degree, degree, dtype=polynomial.dtype, device=polynomial.device)
    companion_matrix[1:, :-1] = torch.eye(degree - 1)
    companion_matrix[0, :] = -polynomial[1:] / polynomial[0]  # Normalize coefficients

    roots = torch.linalg.eigvals(companion_matrix)
    real_roots = roots[roots.isreal()].real.cpu().numpy()

    if max_theta is None:
        max_theta = math.pi / 2

    boundaries = (0, max_theta)
    l_bound, r_bound = boundaries[0], boundaries[1]
    roots_inside_boundaries = real_roots[(l_bound < real_roots) & (real_roots < r_bound)]

    return roots_inside_boundaries.size == 0


def max_theta_for_unmap_from_points(points2d, intrinsics, distortion):
    """ This function returns the (first) max_theta for the provided points """
    normalized = normalize_points2d(points2d, intrinsics)
    r = np.linalg.norm(normalized, axis=-1)

    """

    By definition a fisheye transformation should be monotonous wrt. incident angle
    r_dist_normalized == θ_d
    <=> per monotonous definition
    largest r_dist_normalized <=> largest θ_d <=> largest incident angle θ_original
    THUS: CORNER ==> Largest initial θ!!

    Now, if we find a valid solution θ for this point AND the original function is monotonous up to that point
    It means there should be a valid solution for all θ values!!

    Also this is bounded by θ < 90deg!! because we cannot undistort something with a FOV > 180deg!!
h
    """
    maximum_r = float(r.max())

    # Version with sympy
    # from sympy import symbols, solve # TODO: REQUIREMENTS!
    # s_theta = symbols('theta')
    # k1, k2, k3, k4 = distortion
    # f_theta = s_theta + k1 * s_theta ** 3 + k2 * s_theta ** 5 + k3 * s_theta ** 7 + k4 * s_theta ** 9 - maximum_r
    # solution = solve(f_theta)
    # real_solutions = np.array([np.degrees(float(sol)) for sol in solution if sol.is_real])
    # valid_solutions = ..... # I should only filter based on degree and then return in rad
    # return .....

    # Version with torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    polynomial = torch.zeros(10).to(device)
    polynomial[0] = torch.tensor(distortion[3]).to(device)
    polynomial[2] = torch.tensor(distortion[2]).to(device)
    polynomial[4] = torch.tensor(distortion[1]).to(device)
    polynomial[6] = torch.tensor(distortion[0]).to(device)
    polynomial[8] = 1
    polynomial[9] = -maximum_r

    degree = polynomial.size(0) - 1

    companion_matrix = torch.zeros(degree, degree, dtype=polynomial.dtype, device=polynomial.device)
    companion_matrix[1:, :-1] = torch.eye(degree - 1)
    companion_matrix[0, :] = -polynomial[1:] / polynomial[0]  # Normalize coefficients

    roots = torch.linalg.eigvals(companion_matrix)
    real_roots = roots[roots.isreal()].real.cpu().numpy()  # Keep roots where imaginary part is near zero
    real_solutions_deg = np.degrees(real_roots)
    valid_solutions_within_range = np.sort(real_roots[real_solutions_deg >= 0])  # should be larger than 0!
    if valid_solutions_within_range.size > 0:
        return float(valid_solutions_within_range[0])  # We take the FIRST since we check for monotonicity later
    return None


def intrinsics_from_self(self):
    intrinsics = np.zeros((3, 3), dtype=np.float64)
    intrinsics[0, 0] = self[0]
    intrinsics[1, 1] = self[1]
    intrinsics[0, 2] = self[2]
    intrinsics[1, 2] = self[3]
    intrinsics[2, 2] = 1
    return intrinsics


def distortion_from_self(self):
    return np.array([self[4].numpy(), self[5].numpy(), self[6].numpy(), self[7].numpy()])


class OpenCVFisheye(BaseModel):
    """
    Basically it is the same as simple_radial_fisheye.py
    """
    model_name = 'OPENCV_FISHEYE'
    num_focal_params = 2
    num_pp_params = 2
    num_extra_params = 4

    def __init__(self, x, image_shape):
        super().__init__(x, image_shape)

    @property
    def im_height(self):
        return self[1]  # OTHER WAY THAN NUMPY!!

    @property
    def im_width(self):
        return self[0]  # OTHER WAY THAN NUMPY!!

    def can_unmap_self(self):
        """ This function checks if all of the pixels of the current configuration can be backprojected/unmapped !"""
        intrinsics = intrinsics_from_self(self)
        distortion = distortion_from_self(self)

        # Define grid
        h, w = int(self.im_height), int(self.im_width)
        ud, vd = np.meshgrid(np.arange(0, w), np.arange(0, h))
        points2d = np.stack([ud.ravel(), vd.ravel()], axis=-1)

        # Find max_theta for points
        max_theta = max_theta_for_unmap_from_points(points2d, intrinsics, distortion)  # only returns results > 0deg

        if max_theta is None:
            return False

        theta_as_deg = np.degrees(max_theta).item()
        if theta_as_deg >= 90:
            print("Some pixels would be unmapped with theta>90d.")
            return False

        # Now we need to check if it monotonous up to that maximum
        return is_monotonous_polynomial(distortion, max_theta=max_theta)

    def can_map_self(self):
        intrinsics = intrinsics_from_self(self)
        distortion = distortion_from_self(self)

        # Define grid
        h, w = int(self.im_height), int(self.im_width)
        ud, vd = np.meshgrid(np.arange(0, w), np.arange(0, h))
        points2d = np.stack([ud.ravel(), vd.ravel()], axis=-1)

        # Find max_theta for points
        max_theta = max_theta_for_map_from_points(points2d, intrinsics, h, w)  # only returns results > 0deg

        if max_theta is None:
            return False

        theta_as_deg = np.degrees(max_theta_for_map_from_points(points2d, intrinsics, h, w)).item()
        if theta_as_deg < 0 or theta_as_deg >= 90:
            print("OUTSIDE BOUNDARIES.")  # TODO: remove
            return False

        # Now we need to check if it monotonous up to that maximum
        return is_monotonous_polynomial(distortion, max_theta=min(max_theta, math.pi/2))

    @staticmethod
    def default_initialization(image_shape):
        x = torch.zeros(8)
        x[:2] = torch.tensor([image_shape[0], image_shape[1]]) / 2
        x[2:4] = x[:2]
        return OpenCVFisheye(x, image_shape)

    def map(self, points3d):
        r = torch.norm(points3d[:, :2], dim=-1)
        theta = torch.atan2(r, points3d[:, 2])
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        thetad = theta * (1 + self[4] * theta2 + self[5] * theta4 + self[6] * theta6 + self[7] * theta8)

        uv = torch.zeros_like(points3d[:, :2])
        mask = (r < self.EPSILON) | (theta < self.EPSILON)
        uv[mask] = points3d[:, :2][mask]
        uv[~mask] = points3d[:, :2][~mask] * thetad[:, None][~mask] / r[:, None][~mask]

        return uv * self[:2].reshape(1, 2) + self[2:4].reshape(1, 2), torch.ones_like(r, dtype=torch.bool)

    def unmap(self, points2d):
        uv = (points2d.type(torch.float64) - self[2:4].reshape(1, 2)) / self[:2].reshape(1, 2)

        r = torch.norm(uv, dim=-1)

        polynomials = torch.zeros(r.shape[0], 10).to(r)
        polynomials[:, 9] = self[7]
        polynomials[:, 7] = self[6]
        polynomials[:, 5] = self[5]
        polynomials[:, 3] = self[4]
        polynomials[:, 1] = 1
        polynomials[:, 0] = -r

        theta = NewtonRoot1D.apply(r, polynomials, self.ROOT_FINDING_MAX_ITERATIONS)

        # TODO: REMOVE IF CONFIRMED ;)
        assert np.isclose(
            theta.max().numpy(),
            max_theta_for_unmap_from_points(points2d, intrinsics_from_self(self),  distortion_from_self(self))
        ), "Moritz is dumb"

        mask = (r > self.EPSILON) & (torch.tan(theta) > self.EPSILON)  # This also masks incident angles >90deg
        z = torch.ones_like(r)
        z[mask] = r[mask] / torch.tan(theta[mask])

        return torch.cat((uv, z[..., None]), dim=-1), mask
