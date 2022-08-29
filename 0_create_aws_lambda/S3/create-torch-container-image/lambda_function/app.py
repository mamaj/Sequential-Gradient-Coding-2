# from itertools import islice
# import pickle
# import base64

import torch
import torch.nn.functional as F
# import torchvision


model = torch.load('./vgg16.pt')
x_train, y_train = torch.load('./cifar1000.pt')


# event = {
#     'batch_size': 256,
#     'load': 0.2,
#     'comp_type': 'no_forloop'
# }

def handler(event, contex):
    
    batch_size = int(event['batch_size'])
    load = float(event['load'])
    comp_type = event['comp_type']
    
    n_points = int(load * batch_size)
    _x = x_train[:n_points]
    _y = y_train[:n_points]
    
    
    if comp_type == 'no_forloop':
        _x = _x.unsqueeze(dim=0)
        _y = _y.unsqueeze(dim=0)
        
    elif comp_type == 'forloop':
        _x = _x.unsqueeze(dim=1)
        _y = _y.unsqueeze(dim=1)
        
    
    
    for x, y in zip(_x, _y):
        print(x.shape, y.shape)
        y_logit = model(x)
        loss = F.cross_entropy(y_logit, y)
        
        model.zero_grad()
        loss.backward()
        

    # let us not send back the result. 
    # payload = None
    
    # payload = [p.grad.to(torch.float16) for p in model.parameters()]
    # payload = pickle.dumps(payload, pickle.HIGHEST_PROTOCOL)
    # payload = base64.b64encode(payload)
    
    
    return {
        'statusCode': 200,
        'body': {
            **event,
            'n_points': n_points
        }
    } 
