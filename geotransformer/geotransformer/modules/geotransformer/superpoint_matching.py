import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def torch_farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def forward(self, ref_feats, src_feats, ref_points, src_points,  ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        ref_points = ref_points[ref_indices]
        src_points = src_points[src_indices]        
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        
        # farthest point select       
        ref_points = ref_points[ref_sel_indices]
        src_points = src_points[src_sel_indices]

        torch_fps_ref_indices = self.torch_farthest_point_sample(ref_points.unsqueeze(0), ref_indices.shape[0])
        torch_fps_src_indices = self.torch_farthest_point_sample(src_points.unsqueeze(0), src_indices.shape[0])        
        torch_fps_ref_indices_mask = torch.eq(torch_fps_ref_indices,0)
        torch_fps_src_indices_mask = torch.eq(torch_fps_src_indices,0)

        torch_fps_ref_indices = torch_fps_ref_indices[~torch_fps_ref_indices_mask]
        torch_fps_src_indices = torch_fps_src_indices[~torch_fps_src_indices_mask]
  
        torch_fps_src_indices = torch_fps_src_indices.resize_(1,torch_fps_src_indices.shape[0]+1)
        torch_fps_src_indices = torch_fps_src_indices.squeeze(0)
        torch_fps_src_indices[-1] = 0

        torch_fps_ref_indices = torch_fps_ref_indices.resize_(1,torch_fps_ref_indices.shape[0]+1)
        torch_fps_ref_indices = torch_fps_ref_indices.squeeze(0)
        torch_fps_ref_indices[-1] = 0


        if torch_fps_ref_indices.shape[0] > torch_fps_src_indices.shape[0]:
            fps_indices = torch_fps_ref_indices
        else:
            fps_indices = torch_fps_src_indices

        if fps_indices.shape[0] >= 256:
            torch_src_sel_indices = src_sel_indices[fps_indices[:256]]
            torch_ref_sel_indices = ref_sel_indices[fps_indices[:256]]
        else:
            
            torch_src_sel_indices = src_sel_indices[fps_indices]
            torch_ref_sel_indices = ref_sel_indices[fps_indices]
            fps_range = fps_indices.shape[0]
            src_sel_indices_matrix = src_sel_indices.repeat(fps_range,1)
            src_fps_indices_matrix = torch_src_sel_indices.unsqueeze(1)
            corr_info = src_sel_indices_matrix - src_fps_indices_matrix
            corr_info = torch.eq(corr_info,0).float()
            known_corr_reject = torch.argmax(corr_info,dim=1)
            corr_info[:,known_corr_reject] = 0.0
            corr_info = torch.nonzero(corr_info)
            sel_corr_info = corr_info[:,1]
            sel_corr_sort, _  = sel_corr_info.sort()
            left_sel_indices = ref_sel_indices[sel_corr_sort[:256-fps_range]]
            torch_src_sel_indices = torch_src_sel_indices.resize_(256)
            torch_ref_sel_indices = torch_ref_sel_indices.resize_(256)
            torch_src_sel_indices[fps_range:] = src_sel_indices[sel_corr_sort[:256-fps_range]]
            torch_ref_sel_indices[fps_range:] = ref_sel_indices[sel_corr_sort[:256-fps_range]]


        ref_corr_indices = ref_indices[torch_ref_sel_indices]
        src_corr_indices = src_indices[torch_src_sel_indices]      
        corr_scores = matching_scores[torch_ref_sel_indices,torch_src_sel_indices]
        
        # recover original indices
        # ref_corr_indices = ref_indices[ref_sel_indices]
        # src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores
