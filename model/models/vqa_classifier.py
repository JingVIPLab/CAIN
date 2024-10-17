import sys
import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric

from model.networks.backbone_newidea_4_3 import VQABackbone

import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
   
            
        self.encoder = VQABackbone(args, pre_train=True)                      


    def forward(self, img, que, img_id):
        # out = self.encoder(img, que, img_id)
        logits = self.encoder(img, que, img_id)
        return logits
    
    def forward_proto(self, data_shot, data_query, que_shot, que_query, img_id_shot, img_id_query, way):
        proto = self.encoder(data_shot, que_shot, img_id_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query, que_query, img_id_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim