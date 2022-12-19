from proxyless_nas import model_zoo


def TPNAS_Net(model='proxyless_gpu', pretrained=False):
    """
    Load the searched deep model by using ProxylessNAS algorithm on the 
    ImageNet dataset (or other datasets), but here only provides the 
    method to load the pretrained model (i.e., proxyless_gpu) released 
    by the original paper (Cai et al. 2018).
    Please refer https://github.com/mit-han-lab/ProxylessNAS for more 
    details on the customized searching.
    """
    
    net = model_zoo.__dict__[model](pretrained=pretrained)
    
    return net    

    
    
    