import torch
class FasterRCNN(object):
    ''' detect body
    '''
    def __init__(self, device='cuda:0'):  
        '''
        https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn
        '''
        import torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.5)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds][0].cpu().numpy()
            return bbox
    
    @torch.no_grad()
    def run_multi(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.9)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds].cpu().numpy()
            return bbox