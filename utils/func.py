import torch
import warnings


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


######################################################################
# Save model
#---------------------------

def save_network(model, epoch, acc, save_path):

    print('Saving to', save_path, '...')
    
    if 'module' in dir(model):
        net_save = model.module
    else:
        net_save = model
        
    state = {
        'net': net_save.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'cfg': net_save.cfg
    }

    torch.save(state, save_path)


######################################################################
# Load model
#---------------------------
def load_network(network, net_path):
    
    checkpoint = torch.load(net_path)
    
    if 'net' in checkpoint:
        new_dict = checkpoint['net']
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
    else:
        new_dict = checkpoint
        epoch = -1
        acc = -1
        
    old_dict = network.state_dict()
    
    new_keys = set(new_dict.keys())
    old_keys = set(old_dict.keys())
    
    if old_keys == new_keys:
        network.load_state_dict(new_dict)
        
    else:
        if new_keys-old_keys:
            warnings.warn("Ignoring keys in new dict: {}".format(new_keys-old_keys))        
        if old_keys-new_keys:
            warnings.warn("Missing keys in old dict: {}".format(old_keys-new_keys))

        # filter out unnecessary keys
        new_dict = {k: v for k, v in new_dict.items() if k in old_dict}
        # overwrite entries in the existing state dict
        old_dict.update(new_dict) 
        # load the new state dict
        network.load_state_dict(old_dict)
        
    return network, epoch, acc

