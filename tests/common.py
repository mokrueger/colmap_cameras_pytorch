"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch

def pts2d_pts3d_pts2d_test(camera, points2d):
    points3d = camera.unmap(points2d)
    points2d_new, valid = camera.map(points3d)
    
    diff = points2d_new - points2d
    diff = diff.mean(dim=0)
    return diff, valid.all()

def pts3d_pts2d_pts3d_test(camera, points3d):
    points3d /= torch.norm(points3d, dim=-1, keepdim=True)
    points2d, valid = camera.map(points3d)
    points3d_new = camera.unmap(points2d)
    points3d_new /= torch.norm(points3d_new, dim=-1, keepdim=True)
    diff = points3d_new - points3d
    diff = diff.mean(dim=0)
    return diff, valid.all()

def test_pt2d(model, test_self):
    for u in range(0, model.image_shape[0], 10): 
        for v in range(0, model.image_shape[1], 10):
            pts2d = torch.tensor([[u, v], [u, v], [u, v]]).float()
            diff, valid = pts2d_pts3d_pts2d_test(model, pts2d)
            test_self.assertTrue(valid)
            test_self.assertLess(diff[0].abs().item(), 1e-4)
            test_self.assertLess(diff[1].abs().item(), 1e-4)

def test_pt3d(model, test_self):
    for u in torch.arange(-0.5, 0.5, 0.1):
        for v in torch.arange(-0.5, 0.5, 0.1):
            pts3d = torch.tensor([[u, v, 1.0], [u, v, 1.0], [u, v, 1.0]]).float()
            diff, valid = pts3d_pts2d_pts3d_test(model, pts3d)
            test_self.assertTrue(valid)
            test_self.assertLess(diff[0].abs().item(), 1e-4)
            test_self.assertLess(diff[1].abs().item(), 1e-4)
            test_self.assertLess(diff[2].abs().item(), 1e-4)

def J_scaling(J):
    J_norm = torch.norm(J, dim=-2).mean(dim=-2)
    J_sc = 1 / (1 + J_norm)
    
    J = J * J_sc.unsqueeze(0).unsqueeze(0)
    return J, J_sc

def test_model_fit(model1, model2, iters, test_self):
    data = []
    for u in range(0, model1.image_shape[0], 5): 
        for v in range(0, model1.image_shape[1], 5):
            data.append([u, v])
    points2d = torch.tensor(data).float()
    
    batch_size = points2d.shape[0]
    with torch.no_grad():
        for _ in range(iters):
            def residuals(params_model2):
                model2._data = params_model2
                pts3d = model1.unmap(points2d)
                pts2d, valid = model2.map(pts3d)
                res = points2d[valid] - pts2d[valid]
                return res

            res = residuals(model2._data) 
            J = torch.autograd.functional.jacobian(residuals, model2._data)
            J = J.reshape(batch_size, 2, -1)
            J, J_sc = J_scaling(J)
            
            res = res.reshape(batch_size, 2, 1)
            H = (J.transpose(1, 2) @ J).mean(dim=0)
            b = (J.transpose(1, 2) @ res).mean(dim=0)
            delta = torch.linalg.lstsq(H, b)[0].squeeze()
            delta = delta * J_sc
            model2._data = model2._data - delta

        for u in torch.arange(-0.5, 0.5, 0.1):
            for v in torch.arange(-0.5, 0.5, 0.1):
                points3d = torch.tensor([[u, v, 1.0], [u, v, 1.0], [u, v, 1.0]]).float()
                pts2d_1, valid_1 = model1.map(points3d)
                pts2d_2, valid_2 = model2.map(points3d)
                valid = valid_1 & valid_2
                
                if not valid.any(): continue

                diff = (pts2d_1[valid] - pts2d_2[valid]).abs().mean(dim=0)
                test_self.assertLess(diff[0].abs().item(), 1)
                test_self.assertLess(diff[1].abs().item(), 1)

def test_model_3dpts_fil(model1, model2, test_self):
    torch.autograd.set_detect_anomaly(True)
    data = []
    for u in torch.arange(-0.5, 0.5, 0.1):
        for v in torch.arange(-0.5, 0.5, 0.1):
            data.append([u, v, 1])
    points3d = torch.tensor(data).float()
    points3d /= torch.norm(points3d, dim=-1, keepdim=True) 

    batch_size = points3d.shape[0]
    first_loss = None
    with torch.no_grad():
        for _ in range(5):
            def residuals(params_model2):
                model2._data = params_model2
                pts2d, valid = model1.map(points3d)
                pts3d = model2.unmap(pts2d[valid])
                res = pts3d / torch.norm(pts3d, dim=-1, keepdim=True) - points3d[valid]
                return res

            res = residuals(model2._data) 
            J = torch.autograd.functional.jacobian(residuals, model2._data)
            J = J.reshape(batch_size, 3, -1)
            J, J_sc = J_scaling(J)
            
            res = res.reshape(batch_size, 3, 1)
            H = (J.transpose(1, 2) @ J).mean(dim=0)
            b = (J.transpose(1, 2) @ res).mean(dim=0)
            delta = torch.linalg.lstsq(H, b)[0].squeeze()
            delta = delta * J_sc
            model2._data = model2._data - delta
            
            loss = res.norm(dim=-1).mean()
            if first_loss is None:
                first_loss = loss

    test_self.assertLess(loss, first_loss)
