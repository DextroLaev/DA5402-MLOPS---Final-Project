import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
import config
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class SiameseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048,512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512,config.EMBEDDING_DIM,bias=False),
        )
    
    def forward(self,x):
        out = self.backbone(x)
        embeddings = self.embed(out)
        return F.normalize(embeddings,p=2,dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SiameseEncoder()
    
    def forward_triplet(self,anchor,positive,negative):
        return (self.encoder(anchor),self.encoder(positive),self.encoder(negative))

    def forward(self,img1,img2):
        img1_embed = self.encoder(img1)
        img2_embed = self.encoder(img2)
        dist = F.pairwise_distance(img1_embed,img2_embed,p=2)   #perform l2 distance norm
        return img1_embed,img2_embed,dist

def _init_head(model: SiameseNetwork):
    for m in model.encoder.embed.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def build_model():
    model = SiameseNetwork()
    _init_head(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Resnet50 Model | total params: {total_params/1e6:.1f}M  trainable={trainable_params/1e6:.1f}M')
    return model