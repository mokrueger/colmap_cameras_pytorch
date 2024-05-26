"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
import argparse

from colmap_cameras import model_selector, default_initialization

arg_parser = argparse.ArgumentParser("Refit one colmap model to another")
input_camera_str = arg_parser.add_argument("--input_camera", type=str, help="Input camera model in colmap foramt. Example: PINHOLE 100 100 100 100 50 50")
output_camera_name = arg_parser.add_argument("--output_camera", type=str, help="Output camera model in colmap format. Example: SIMPLE_RADIAL")
iterations = arg_parser.add_argument("--iterations", type=int, default=20, help="Number of iterations for optimization")

args = arg_parser.parse_args()
input_camera = args.input_camera
output_camera = args.output_camera

input_camera_name = input_camera.split()[0]
input_camera_data = list(map(float, input_camera.split()[1:]))

input_camera = model_selector(input_camera_name, input_camera_data)
output_camera = default_initialization(output_camera, input_camera.image_shape)

print("Input model : \n\t", input_camera.to_colmap())
print("Output model init : \n\t", output_camera.to_colmap())

full_data = []
for x in range(int(input_camera.image_shape[0])):
    for y in range(int(input_camera.image_shape[1])):
        full_data.append([x, y])

full_data = torch.tensor(full_data, dtype=torch.float32)     

iterations = args.iterations
batch_size = 2000
print("#"*60)
for i in range(iterations):
    H_acc = torch.zeros((output_camera._data.shape[0], output_camera._data.shape[0])).to(full_data)
    b_acc = torch.zeros((output_camera._data.shape[0], 1)).to(full_data)
    
    err_acc = 0
    num_adds = 0
    for j in range(0, full_data.shape[0], batch_size):
        num_adds += 1
        data = full_data[j:j+batch_size]
        def res(output_camera_data):
            output_camera._data = output_camera_data
            pts3d = input_camera.unmap(data)
            pts2d, valid = output_camera.map(pts3d)

            return pts2d[valid] - data[valid]
        
        J = torch.autograd.functional.jacobian(res, output_camera._data)
        err = res(output_camera._data)
        J = J.reshape(batch_size, 2, -1)

        H_acc += (J.transpose(1, 2) @ J).mean(dim=0)
        b_acc += (J.transpose(1, 2) @ err[...,None]).mean(dim=0)

        err_acc += err.norm(dim=-1).mean()

    H_acc /= num_adds
    b_acc /= num_adds

    err_acc /= num_adds

    step = torch.linalg.lstsq(H_acc, b_acc)[0].squeeze()
    output_camera._data = output_camera._data - step
    print(f'Iteration {i} : Mean reprojection error {err_acc.item()}')
print("#"*60)
print("FINAL MODEL:")
print("\t", output_camera.to_colmap())
            
