from avatar3d import *
from torch_dataset import *
from torch.utils.data import DataLoader
import os
import os.path as osp
from smplx.smplx.body_models import *
from numpy.linalg import norm
from utils import *
import json

def test(data, 
        image_path, 
        smplx_model_path, 
        trained_model_path,
        result_path=None, 
        batch_size=1, 
        device='cuda:0', 
        print_loss=50,
        show_results=False):

    if result_path != None and not osp.exists(result_path):
        os.mkdir(result_path)
    
    checkpoint = torch.load(trained_model_path)
    model = Avatar3D()
    model.load_state_dict(checkpoint['model_state_dict'])

    test_set = ImageDataset(data, image_path, mode='test')
    test_dataloader = DataLoader(test_set, batch_size=batch_size)
    loss_fn = nn.MSELoss()
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        running_loss = 0.
        vertices_distance = 0.
        joints_distance = 0.
        for i, batch in enumerate(test_dataloader):
            img, keypoints, pose = batch['image'].to(device), batch['keypoints'].to(device), batch['pose'].to(device)
            output = model(img, keypoints)
            loss = loss_fn(output, pose)
            running_loss += loss.item()
            smplx_gt_model = create(smplx_model_path, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         num_pca_comps=15
                         )
            smplx_pred_model = create(smplx_model_path, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         num_pca_comps=15
                         )
            smplx_gt_model.to(device)
            smplx_pred_model.to(device)
            #output = output.cpu()
            #pose = pose.cpu()
            gt_output = smplx_gt_model(return_verts=True, create_body_pose = False, body_pose = pose)
            gt_vertices = gt_output.vertices.detach().cpu().numpy().squeeze()
            gt_joints = gt_output.joints.detach().cpu().numpy().squeeze()
            pred_output = smplx_pred_model(return_verts=True, create_body_pose = False, body_pose = output)
            pred_vertices = pred_output.vertices.detach().cpu().numpy().squeeze()
            pred_joints = pred_output.joints.detach().cpu().numpy().squeeze()
            vertices_distance += norm(gt_vertices - pred_vertices)
            joints_distance += norm(gt_joints - pred_joints)

            result = {'image_path':batch['image_path'],
                      'pred_vertices':pred_vertices.tolist(),
                      'pred_joints':pred_joints.tolist(),
                      'gt_vertices':gt_vertices.tolist(),
                      'gt_joints':gt_joints.tolist() 
                     }
            results.append(result)

            if show_results:
                show_3d_human(smplx_pred_model, pred_vertices, pred_joints)

            if i % print_loss == 0:
                print("idx: %d"%i)
                print("loss: %f"%(running_loss/(i+1)))
                print("vertices_distance: %f"%(vertices_distance/(i+1)))
                print("joints_distance: %f"%(joints_distance/(i+1)))
            if i == 50:
                break
        
        avg_loss = running_loss/(i+1)
        avg_vertices_distance = vertices_distance/(i+1)
        avg_joints_distance = joints_distance/(i+1)
        f = open(osp.join(result_path, 'result.json'), 'w')
        json.dump(results, f, indent=4)
        f.close()
        
        print("avg_loss: %f"%avg_loss)
        print("avg_vertices_distance: %f"%avg_vertices_distance)
        print("avg_joints_distance: %f"%avg_joints_distance)
