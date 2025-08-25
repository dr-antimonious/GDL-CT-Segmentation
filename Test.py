from collections import OrderedDict

from torch import load, device

from Network import CHD_GNN

DIRECTORY = "/home/ubuntu/proj/GDL-CT-Segmentation/MODELS/"

def main():
    dev = device("cuda:0")
    model = CHD_GNN().to(dev)
    checkpoint = load(DIRECTORY + "gnn_90.checkpoint", map_location = dev)["MODEL_STATE"]
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        nk = k.replace("module.", "")
        new_state_dict[nk] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
        
if __name__ == '__main__':
    main()