import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from classification.classification_model import classification_model

from geotransformer.modules.kpconv.modules import GlobalAvgPool
from backbone import KPConvFPN

import random
from torch.autograd import Variable

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        for p in self.parameters():
            p.requires_grad=False
        self.classification_model = classification_model()          

    def forward(self, data_dict):
        output_dict = {}
        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_points_c, src_points_c, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # classification data prepare
        classification_one_data_dict = classification_data_prepare(self.training, ref_points_c.shape[0], src_points_c.shape[0], gt_node_corr_overlaps.detach(), gt_node_corr_indices.detach(), output_dict['ref_node_corr_indices'].detach(), output_dict['src_node_corr_indices'].detach(), ref_feats_c_norm.detach(), src_feats_c_norm.detach())
        classification_inputs = Variable(classification_one_data_dict['corr_node_feat'],requires_grad=True).to('cuda')
        classification_ground_truth = Variable(classification_one_data_dict['ground_truth']).to('cuda')
        predict_results = self.classification_model(classification_inputs)
        output_dict['predict_results'] = predict_results
        output_dict['classification_ground_truth'] = classification_ground_truth.detach()

        if not self.training:
            # sorted_values, sorted_indices = torch.sort(predict_results.squeeze(),descending=True)         
            predict_results1 = torch.gt(predict_results.squeeze(), 0.80)
            # print("predict_results:",predict_results)
            predict_results1 = torch.nonzero(predict_results1).squeeze()
            # print("predict_results1:",predict_results1.numel())
            ref_node_corr_indices1 = ref_node_corr_indices[predict_results1]
            src_node_corr_indices1 = src_node_corr_indices[predict_results1]
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices1
            output_dict['src_node_corr_indices'] = src_node_corr_indices1 
            
            if  predict_results1.numel() <= 20:
                predict_results2 = torch.gt(predict_results.squeeze(), 0.55)
                # print("predict_results:",predict_results)
                predict_results2 = torch.nonzero(predict_results2).squeeze()
                # print("predict_results:",predict_results)
                ref_node_corr_indices2 = ref_node_corr_indices[predict_results2]
                src_node_corr_indices2 = src_node_corr_indices[predict_results2]
                output_dict['ref_node_corr_indices'] = ref_node_corr_indices2
                output_dict['src_node_corr_indices'] = src_node_corr_indices2  
            ref_node_corr_indices = output_dict['ref_node_corr_indices']
            src_node_corr_indices = output_dict['src_node_corr_indices']       

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores   

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform
            output_dict['classification_result'] = 0 #classify2
        return output_dict

def classification_data_prepare(mode, ref_length_c, src_length_c, gt_node_corr_overlaps, gt_node_corr_indices, ref_node_corr_indices, src_node_corr_indices, ref_feats_c_norm, src_feats_c_norm):
    # ref_length_c = data_dict['ref_points_c'].shape[0]
    # src_length_c = data_dict['src_points_c'].shape[0]
    # gt_node_corr_overlaps = data_dict['gt_node_corr_overlaps'].data
    # gt_node_corr_indices = data_dict['gt_node_corr_indices'].data
    one_data = {}
    masks = torch.gt(gt_node_corr_overlaps, 0.0)
    gt_node_corr_indices = gt_node_corr_indices[masks]
    gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
    gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
    gt_node_corr_map = torch.zeros(ref_length_c, src_length_c)
    gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

    # ref_node_corr_indices = data_dict['ref_node_corr_indices'].data
    # src_node_corr_indices = data_dict['src_node_corr_indices'].data

    corr_node_ground_truth = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices]       

    # if self.mode == 'train':     
    ground_truth_pos = torch.nonzero(gt_node_corr_map)
    ground_truth_neg = torch.nonzero(torch.eq(gt_node_corr_map,0))
    pos_index = [i for i in range(ground_truth_pos.shape[0])]
    random.shuffle(pos_index)
    ground_truth_pos = ground_truth_pos[pos_index[:2500],:]

    neg_index = [i for i in range(ground_truth_neg.shape[0])]
    random.shuffle(neg_index)
    ground_truth_neg = ground_truth_neg[neg_index[:ground_truth_pos.shape[0]],:]    

    ground_truth_both = torch.cat((ground_truth_pos,ground_truth_neg),dim=0)

    random_index = [i for i in range(ground_truth_both.shape[0])]
    random.shuffle(random_index)
    ground_truth_both = ground_truth_both[random_index]
    if mode == 'training':
        ref_node_corr_indices = ground_truth_both[:,0]
        src_node_corr_indices = ground_truth_both[:,1]

    corr_node_ground_truth = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices]

    # ref_feats_c_norm = data_dict['ref_feats_c'].data
    # src_feats_c_norm = data_dict['src_feats_c'].data

    ref_corr_node_feats = ref_feats_c_norm[ref_node_corr_indices]
    src_corr_node_feats = src_feats_c_norm[src_node_corr_indices]

    mean = src_corr_node_feats.mean()
    var = src_corr_node_feats.var()
    src_corr_node_feats = (src_corr_node_feats - mean) / torch.pow(var + 1e-05,0.5)  

    mean = ref_corr_node_feats.mean()
    var = ref_corr_node_feats.var()
    ref_corr_node_feats = (ref_corr_node_feats - mean) / torch.pow(var + 1e-05,0.5)  

    # print("src_corr_node_feate:",src_corr_node_feate.shape)
    # print("ref_corr_node_feats:",ref_corr_node_feats.shape)


    corr_node_feat = torch.cat((ref_corr_node_feats.unsqueeze(0).transpose(0,1), src_corr_node_feats.unsqueeze(0).transpose(0,1)), dim=1)

    corr_node_feat = corr_node_feat.repeat(1,1,2)

    corr_node_feat = corr_node_feat.chunk(16,dim=2)
    # print("corr_node_feat:",corr_node_feat.shape)
    corr_node_feat = torch.cat((corr_node_feat),dim=1)  

    corr_node_feat = corr_node_feat.unsqueeze(1) 


    one_data['corr_node_feat'] = corr_node_feat
    one_data['ground_truth'] = corr_node_ground_truth.unsqueeze(1)   
    return one_data

def create_model(cfg):
    model = GeoTransformer(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
