from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device=None):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k, device=device) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class TranformerEncoder(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, num_layers, device=None):
        super(TranformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output


class TemporalTransformer(nn.Module):
    def __init__(
            self,
            input_v_2d_dim,
            input_v_3d_dim,
            input_a_dim,
            hidden_dim,
            nhead,
            ff_dim,
            num_layers,
            dropout=0.1,
            pre_norm=False,
            clap_event_feats=None,
            clip_event_feats=None,
            device=None
    ):
        super(TemporalTransformer, self).__init__()

        self.input_v_2d_dim = input_v_2d_dim   # 2048: ResNet152, 768: CLIP large
        self.input_v_3d_dim = input_v_3d_dim
        self.input_a_dim = input_a_dim         # 128: VGGish, 512: CLAP
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pre_norm = pre_norm

        self.proj_a = MLP(self.input_a_dim, self.hidden_dim*2, self.hidden_dim, num_layers=2, device=device)
        self.proj_v = MLP(self.input_v_2d_dim, self.hidden_dim*2, self.hidden_dim, num_layers=2, device=device)

        self.a_transformer = TranformerEncoder(self.hidden_dim, self.nhead, dim_feedforward=self.ff_dim,
                                                dropout=self.dropout, num_layers=self.num_layers, device=device)
        self.v_transformer = TranformerEncoder(self.hidden_dim, self.nhead, dim_feedforward=self.ff_dim,
                                                dropout=self.dropout, num_layers=self.num_layers, device=device)

        self.clap_event_feats = clap_event_feats    # L2 normalized
        self.clip_event_feats = clip_event_feats    # L2 normalized
        self.register_buffer('a_logit_scale', torch.tensor(3.2912, dtype=torch.float32, device=device))
        self.register_buffer('v_logit_scale', torch.tensor(4.6052, dtype=torch.float32, device=device))

        self.final_proj_a = MLP(self.hidden_dim, 512, 512, num_layers=2, device=device)
        self.final_proj_v = MLP(self.hidden_dim, 768, 768, num_layers=2, device=device)

    def forward(self, clap_feat, clip_feat, valid_mask=None, attn_mask=None, src_key_padding_mask=None):

        a_feat = self.proj_a(clap_feat)
        v_feat = self.proj_v(clip_feat)

        a_feat = rearrange(a_feat, 'b t d -> t b d')
        v_feat = rearrange(v_feat, 'b t d -> t b d')
        if attn_mask is not None:
            attn_mask = ~attn_mask  # change values of pad position from False to True
            attn_mask = attn_mask.to(torch.int32).to(torch.float32) * (-1e10)   # not using bool mask to avoid nan
            attn_mask = repeat(attn_mask, 'b l s -> (b h) l s', h=self.nhead)
        a_feat = self.a_transformer(a_feat, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        v_feat = self.v_transformer(v_feat, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        a_feat = rearrange(a_feat, 't b d -> b t d')
        v_feat = rearrange(v_feat, 't b d -> b t d')


        a_feat = self.final_proj_a(a_feat)
        v_feat = self.final_proj_v(v_feat)
        a_logit = torch.matmul(a_feat, self.clap_event_feats.transpose(0, 1)) * self.a_logit_scale
        v_logit = torch.matmul(v_feat, self.clip_event_feats.transpose(0, 1)) * self.v_logit_scale

        return a_logit, v_logit
    
    def calculate_loss(self, args, outputs, labels, valid_masks, logits=None, thresholds=None):

        a_logits, v_logits = outputs
        B, T, C = a_logits.shape

        gt_av_labels = labels   # (B, T, C)
            
        loss_video = torch.tensor(0.0, device=a_logits.device, dtype=torch.float32)
            
        av_probs = a_logits.sigmoid() * v_logits.sigmoid()    # (B, T, C)
        loss_av = F.binary_cross_entropy(av_probs, gt_av_labels, reduction='none')  # (B, T, C)
        class_weight = torch.where(gt_av_labels.sum(dim=1, keepdim=True) > 0, args.pos_weight, args.neg_weight)
        loss_av = loss_av * class_weight
        loss_av = loss_av * valid_masks.unsqueeze(dim=-1)
        loss_av = loss_av.sum() / (valid_masks.sum() * C)

        loss = loss_video + loss_av

        loss_dict = {
            'loss_video': loss_video.item(),
            'loss_av': loss_av.item(),
            'loss_all': loss.item()
        }
        return loss, loss_dict
