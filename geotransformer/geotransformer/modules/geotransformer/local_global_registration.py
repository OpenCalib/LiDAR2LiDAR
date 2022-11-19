from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform
from geotransformer.modules.registration import WeightedProcrustes
from torch.autograd import Variable
import torch.nn.functional as F
# import sys
# path = r"GeoTransformer-corr-classification-testRR/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/classification.py"
# sys.path.append(path)
# import classification
class Resnet(nn.Module):
    def __init__(self,basicBlock,blockNums,nb_classes):
        super(Resnet, self).__init__()
        self.in_planes=64
        #输入层
        self.conv1=nn.Conv2d(1,self.in_planes,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layers(basicBlock,blockNums[0],64,1)
        self.layer2=self._make_layers(basicBlock,blockNums[1],128,2)
        self.layer3=self._make_layers(basicBlock,blockNums[2],256,2)
        self.layer4=self._make_layers(basicBlock,blockNums[3],512,2)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(512,nb_classes)

    def _make_layers(self,basicBlock,blockNum,plane,stride):
        """

        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param plane: 输出通道数
        :param stride: 卷积步长
        :return:
        """
        layers=[]
        for i in range(blockNum):
            if i==0:
                layer=basicBlock(self.in_planes,plane,3,stride=stride)
            else:
                layer=basicBlock(plane,plane,3,stride=1)
            layers.append(layer)
        self.in_planes=plane
        return nn.Sequential(*layers)
    def forward(self,inx):
        x=self.maxpool(self.relu(self.bn1(self.conv1(inx))))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.shape[0],-1)
        out=self.fc(x)
        return out

class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""
    def __init__(self,in_planes,planes,kernel_size=3,stride=1):
        """

        :param in_planes: 输入通道
        :param planes:  输出通道
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        """
        super(basic_block, self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=kernel_size,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(planes,planes,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        if stride!=1 or in_planes!=planes:
            self.downsample=nn.Sequential(nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride)
                                          ,nn.BatchNorm2d(planes))
        else:
            self.downsample=nn.Sequential()
    def forward(self,inx):
        x=self.relu(self.bn1(self.conv1(inx)))
        x=self.bn2(self.conv2(x))
        out=x+self.downsample(inx)
        return F.relu(out)

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.resnet18=Resnet(basic_block,[2,2,2,2],256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 1) 
        self.activate1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.activate2 = nn.Sigmoid()
 
    def forward(self, x):
        x = self.resnet18(x)
        x = self.linear1(x)
        x = self.activate1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.activate2(x)
        return x


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)


    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)
        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
        corr_mat = torch.logical_and(corr_mat, mask_mat)

        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat, 
    #ref_node_corr_knn_feats, src_node_corr_knn_feats, transform
    ):
        # extract dense correspondences
        # batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        # global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        # global_src_corr_points = src_knn_points[batch_indices, src_indices]
        # global_ref_corr_points_feat = ref_node_corr_knn_feats[batch_indices, ref_indices]
        # global_src_corr_points_feat = src_node_corr_knn_feats[batch_indices, src_indices]        
        # global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        
        # classify_model = torch.load('/Download/GeoTransformer-corr-classification-testRR/GeoTransformer/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/classification.pth')
        # classify_model.to('cuda')
        # classify_model.eval()  

        # global_src_corr_points = apply_transform(global_src_corr_points, transform)
        # residuals = torch.sqrt(((global_ref_corr_points - global_src_corr_points) ** 2) .sum(1))
        # residuals = torch.tensor(residuals)
        # ground_truth = torch.lt(residuals , 0.1).float().unsqueeze(1)

        # mean = global_src_corr_points_feat.mean()
        # var = global_src_corr_points_feat.var()
        # src_mean_var_text = "src_mean:{:.5f}, src_var:{:.5f}".format(mean, var)
        # global_src_corr_points_feat = (global_src_corr_points_feat - mean) / torch.pow(var + 1e-05,0.5)  
        # mean = global_ref_corr_points_feat.mean()
        # var = global_ref_corr_points_feat.var()
        # ref_mean_var_text = "  ref_mean:{:.5f}, ref_var:{:.5f}   ".format(mean, var)
        # global_ref_corr_points_feat = (global_ref_corr_points_feat - mean) / torch.pow(var + 1e-05,0.5)  

        # # with open("geo_all_middle_text.txt","a+") as f:
        # #     f.write(src_mean_var_text + ref_mean_var_text)


        # global_src_corr_points_feat = global_src_corr_points_feat.unsqueeze(0)
        # global_ref_corr_points_feat = global_ref_corr_points_feat.unsqueeze(0)

        # corr_points_feat = torch.cat((global_ref_corr_points_feat.transpose(0,1), global_src_corr_points_feat.transpose(0,1)), dim=1)
        # corr_points_feat = corr_points_feat.repeat(1,1,2)
        # corr_points_feat = corr_points_feat.chunk(16,dim=2)
        # corr_points_feat = torch.cat((corr_points_feat),dim=1)                        

        # inputs = Variable(corr_points_feat.unsqueeze(1)).to('cuda')
        # target = Variable(ground_truth).to('cuda').squeeze(0)
        # out = classify_model(inputs) # 前向传播
        # predict = torch.gt(out, 0.5).float()

        # total_corr_numbers = predict.shape[0]
        # Inliers_number = target.cpu().clone().detach().squeeze(1).sum()
        # Outliers_number = target.cpu().clone().detach().squeeze(1).shape[0] - Inliers_number
        # predict_1_unmber = predict.cpu().clone().detach().squeeze(1).sum()
        # predict_0_unmber = predict.cpu().clone().detach().squeeze(1).shape[0] - predict_1_unmber
        # predict_1_and_True_unmber = torch.eq(predict.cpu().clone().detach().squeeze(1) + target.cpu().clone().detach().squeeze(1),2).float().sum()
        # predict_1_and_False_unmber = predict_1_unmber - predict_1_and_True_unmber
        # predict_0_and_True_unmber = torch.eq(predict.cpu().clone().detach().squeeze(1) + target.cpu().clone().detach().squeeze(1),0).float().sum()
        # predict_0_and_False_unmber = predict_0_unmber - predict_0_and_True_unmber 
        # print("gt_IR:", Inliers_number/total_corr_numbers) 
        # print("predict_IR:", predict_1_and_True_unmber/predict_1_unmber) 
        # print("predict_Acc:", (predict_1_and_True_unmber+predict_0_and_True_unmber)/total_corr_numbers) 

        # with open("test_origin3.txt","a+") as f:
        #     txtstr = 'gt_IR:{:<.5f},  predict_IR:{:<.5f}, predict_Acc:{:<.5f}, gt_Inliers:{:<.0f}, gt_Outliers:{:<.0f},  predict_1_unmber:{:<.0f},  predict_1_and_True_unmber:{:<.0f},  predict_1_and_False_unmber:{:<.0f},  predict_0_unmber:{:<.0f},  predict_0_and_True_unmber:{:<.0f}, predict_0_and_False_unmber:{:<.0f} \n'.format(Inliers_number/total_corr_numbers, predict_1_and_True_unmber/predict_1_unmber, \
        #             (predict_1_and_True_unmber+predict_0_and_True_unmber)/total_corr_numbers, \
        #             Inliers_number, Outliers_number,  \
        #             predict_1_unmber, predict_1_and_True_unmber, predict_1_and_False_unmber, \
        #             predict_0_unmber, predict_0_and_True_unmber, predict_0_and_False_unmber    )
        #     f.write(txtstr)

        # corr_mat[batch_indices, ref_indices, src_indices] = (predict == 1).squeeze()

        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        # global_ref_corr_points_feat = ref_node_corr_knn_feats[batch_indices, ref_indices]
        # global_src_corr_points_feat = src_node_corr_knn_feats[batch_indices, src_indices]        
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
            # ref_corr_points_feat = global_ref_corr_points_feat[sel_indices]
            # src_corr_points_feat = global_src_corr_points_feat[sel_indices]            
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            # ref_corr_points_feat = global_ref_corr_points_feat
            # src_corr_points_feat = global_src_corr_points_feat             
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform, #global_ref_corr_points_feat, global_src_corr_points_feat  

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
        # ref_node_corr_knn_feats,
        # src_node_corr_knn_feats,
        # transform
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat, #ref_node_corr_knn_feats, src_node_corr_knn_feats, transform
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform#, corr_mat, ref_corr_points_feat, src_corr_points_feat
