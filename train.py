from avatar3d import *
from torch_dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep
import os
import os.path as osp

def train(
    data, 
    data_path,
    batch_size=32, 
    epoch = 1000, 
    learning_rate=1e-4, 
    momentum=0.1, 
    print_loss=50, 
    save_model=True,
    save_freq=1000, 
    save_path='./checkpoints',
    device='cuda:0'):
    
    save_count = 0
    if not osp.exists(save_path):
        os.mkdir(save_path)

    train_set = ImageDataset(data, data_path, mode='train')
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    model = Avatar3D()
    model = model.to(device)
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    pbar = tqdm(total = epoch)
    epoch_loss = 0.
    for e in range(epoch):
        running_loss = 0.    
        for i, batch in enumerate(train_dataloader):
            img, keypoints, pose = batch['image'].to(device), batch['keypoints'].to(device), batch['pose'].to(device)
            optimizer.zero_grad()
            output = model(img, keypoints)
            loss = loss_fn(output, pose)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #sleep(0.01)
        avg_batch_loss = running_loss/(i+1)
        epoch_loss += avg_batch_loss
        if e % print_loss == 0:
            print("epoch:%d, running loss: %f" % (e,epoch_loss/(e+1)))
            
        if save_model and e % save_freq == 0:
            save_count += 1
            torch.save({
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/(e+1),
                'epoch': e,
                'batch_size': batch_size
            }, osp.join(save_path, f'checkpoint%d.pth'%save_count))
        pbar.update()

    torch.save({
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/(e+1),
                'epoch': e,
                'batch_size': batch_size
            }, osp.join(save_path, f'checkpoint_final.pth'))
    return model






