import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class OffDetector(nn.Module):

    def __init__(self, hidden_size=1280):
        super().__init__()

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280,hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size,1)
            )

    def forward(self, x):
        ## Classifies N SUBBURSTS into one logit (for each in batch)

        bs, sb, nc = x.size()[:3]

        # BATCH X SUB_BURSTS X CROPS X CH X H X W -> BATCH X SUB_BURSTS X CROPS-1 X CH X H X W
        diffs = x[:,:,:-1] - x[:,:,1:]

        # BATCH X SUB_BURSTS X CROPS X CH X H X W -> BATCH*SUB_BURSTS*(CROPS-1) X CH X H X W
        diffs = diffs.view(bs*sb*(nc-1),*diffs.size()[3:])

        # BATCH*SUB_BURSTS*(CROPS-1) X CH X H X W -> BATCH*SUB_BURSTS*(CROPS-1) X EMB_SIZE
        feats_diff = self.encoder.extract_features(diffs)
        feats_diff = self.avg_pooling(feats_diff).squeeze()

        # BATCH*SUB_BURSTS*(CROPS-1) X EMB_SIZE -> BATCH*SUB_BURSTS*(CROPS-1) X 1
        out = self.classifier(feats_diff)

        # BATCH*SUB_BURSTS*(CROPS-1) X 1 -> BATCH X SUB_BURSTS
        out = out.view(bs,sb,nc-1)
        out = out.mean(2)

        # BATCH X SUB_BURSTS -> BATCH
        out = out.mean(1)

        return out
