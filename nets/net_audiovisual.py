import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pdb


class LabelSmoothingNCELoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.log(torch.sum(true_dist * pred, dim=self.dim)))


class IntraEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512, nhead=1, dim_feedforward=512, dropout=0.1):
        super(IntraEncoder, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(IntraHANLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout))

        self.layers = nn.ModuleList(self.layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None, with_ca=True):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            src_a = output_a
            src_v = output_v

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class IntraHANLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(IntraHANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src_q: the sequence to the encoder layer (required).
            src_v: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            with_ca: whether to use audio-visual cross-attention
        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if with_ca:
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)

        src_q = src_q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)



class NREP(nn.Module):

    def __init__(self, num_layers=1, temperature=0.2, att_dropout=0.1, cls_dropout=0.5, warm_num=1000, dim_a = 768, dim_v = 768, dim_st = 512):
        super(NREP, self).__init__()

        self.warm_num = warm_num

        self.fc_a_sl = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        self.fc_v_sl = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_frame_att = nn.Linear(512, 25 * 2)
        self.fc_temporal_att = nn.Linear(512, 25 * 2)

        self.fc_a = nn.Sequential(
            nn.Linear(dim_a, 512),
            nn.ELU(),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(dim_v, 512),
            nn.ELU(),
        )
        self.fc_st = nn.Sequential(
            nn.Linear(dim_st, 512),
            nn.ELU(),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
        )

        self.inter_encoder = IntraEncoder('HANLayer', num_layers, norm=None, d_model=512, nhead=1, dim_feedforward=512, dropout=att_dropout)

        self.fc_joint_sl = nn.Sequential(
            nn.Linear(512, 25*2),
            )
        self.fc_sl_fore = nn.Sequential(
            nn.Linear(512, 25),
            )
        self.fc_sl_back = nn.Sequential(
            nn.Linear(512, 25),
            )

        self.temp = temperature
        if cls_dropout != 0:
            self.dropout = nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None


    def forward(self, audio, visual, visual_st, with_ca=True, epoch_num = 0):
        
        b, t, d = visual_st.size()
        m = 2
        c = 25
        # x1 = self.fc_a(audio)
        if audio.size(1) == 64:     # input data are feature maps
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, 768, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, 512, 10)
            x1 = x1.permute(0, 2, 1)
        else:
            x1 = self.fc_a(audio)
        
        # 2d and 3d visual feature fusion
        if visual.size(1) == 80:        # input 2d features are from ResNet152
            vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
            vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        else:                           # input 2d features are from CLIP
            vid_s = self.fc_v(visual)

        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)


        intra_x1, intra_x2 = self.inter_encoder(x1, x2, with_ca=False)
        inter_x1, inter_x2 = self.inter_encoder(x1, x2, with_ca=True)
        ratio_a = 0.8
        ratio_v = 0.8
        x1 = intra_x1 * ratio_a + inter_x1 * (1 - ratio_a)
        x2 = intra_x2 * ratio_v + inter_x2 * (1 - ratio_v)


        # noise contrastive
        # please refer to https://github.com/Yu-Wu/Modaily-Aware-Audio-Visual-Video-Parsing
        xx1_after = F.normalize(x1, p=2, dim=-1)
        xx2_after = F.normalize(x2, p=2, dim=-1)
        sims_after = xx1_after.bmm(xx2_after.permute(0, 2, 1)).squeeze(1) / self.temp
        sims_after = sims_after.reshape(-1, 10)
        mask_after = torch.zeros(b, 10)
        mask_after = mask_after.long()
        for i in range(10):
            mask_after[:, i] = i
        mask_after = mask_after.to(x1.device)
        mask_after = mask_after.reshape(-1)

        # prediction
        if self.dropout is not None:
            inter_x2 = self.dropout(inter_x2)

        pred_fore = torch.cat([inter_x1.unsqueeze(-2), inter_x2.unsqueeze(-2)], dim=-2)
        pred_back = torch.cat([intra_x2.unsqueeze(-2), intra_x1.unsqueeze(-2)], dim=-2)
        x = torch.cat([inter_x1.unsqueeze(-2), inter_x2.unsqueeze(-2)], dim=-2)



        frame_av_va = self.fc_sl_fore(pred_fore)  # 128, 10, 2, 25
        frame_v_a = self.fc_sl_back(pred_back)  # 128, 10, 2, 25
        frame_logits = torch.stack((frame_av_va, frame_v_a), dim=-1) # 128, 10, 2, 25, 2


        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x).reshape(b, t, m, c, 2), dim=1)

        attn_frame_logits = frame_att * frame_logits
        a_logits = attn_frame_logits[:, :, 0, :, :].sum(dim=1)
        v_logits = attn_frame_logits[:, :, 1, :, :].sum(dim=1)

        x1_fusion = self.fc_a_sl(x1)
        x2_fusion = self.fc_v_sl(x2)
        temporal_logits = self.fc_joint_sl(x1_fusion + x2_fusion).reshape(b, t, 25, 2) # (B, T, C)

        temporal_att_before_softmax = self.fc_temporal_att(x1_fusion + x2_fusion).reshape(b, t, c, 2)
        temporal_att = torch.softmax(temporal_att_before_softmax, dim=1)
        fore_global_logits = (temporal_logits[...,0]*temporal_att[...,0]).sum(dim=1) # (B, C)
        back_global_logits = (temporal_logits[...,1]*temporal_att[...,1]).sum(dim=1) # (B, C)

        global_logits = torch.stack((fore_global_logits, back_global_logits), dim=-1) # (B, C, 2)


        return global_logits, a_logits, v_logits, frame_logits, temporal_logits, frame_att, temporal_att, sims_after, mask_after

