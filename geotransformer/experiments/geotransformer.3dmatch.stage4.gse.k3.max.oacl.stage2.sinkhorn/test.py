import argparse
import os.path as osp
import time
import torch
import numpy as np
import random

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader,train_valid_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator

from torch.autograd import Variable
import torch.nn as nn
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        # data_loader, neighbor_limits = train_valid_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']
        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        # with open("test_origin2.txt","a+") as f:
        #     f.write(file_name) 
        #     f.write("\n")        
        # with open("geo_all_middle_text1.txt","a+") as f:
        #     f.write(file_name) 
        #     f.write("\n")        
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            # corr_mat=release_cuda(output_dict['corr_mat']),
            # ref_corr_points_feat=release_cuda(output_dict['ref_corr_points_feat']),
            # src_corr_points_feat=release_cuda(output_dict['src_corr_points_feat']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )

    # def step_scheduler_and_save_model(self):        
    #     self.scheduler.step()     
    #     torch.save(self.classification_model, '/dssg/home/acct-eeyj/eeyj-user1/WPJ/GeoTransformer-nodecorr-classification/GeoTransformer/output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/node_classification.pth')     


    # def train_classification_model(self,output_dict):
    #     one_data_dict = self.classification_dataset_prepare(output_dict)
    #     inputs = Variable(one_data_dict['corr_node_feat'],requires_grad=True).to('cuda')
    #     target = Variable(one_data_dict['ground_truth'],requires_grad=True).to('cuda')
    #     print("inputs:",inputs.shape)  
    #     print("target:",target.shape) 
    #     out = self.classification_model(inputs) # 前向传播
    #     out = out.squeeze()
    #     loss = self.criterion(out, target) # 计算误差
    #     loss.requires_grad_(True)
    #     self.optimizer.zero_grad() # 梯度清零
    #     loss.backward() # 后向传播
    #     self.optimizer.step() # 调整参数      
    #     predict = torch.gt(out, 0.5).float()
    #     Inliers_number = target.cpu().clone().detach().sum()
    #     Outliers_number = target.cpu().clone().detach().shape[0] - Inliers_number
    #     predict_1_unmber = predict.cpu().clone().detach().sum()
    #     predict_0_unmber = predict.cpu().clone().detach().shape[0] - predict_1_unmber
    #     predict_1_and_True_unmber = torch.eq(predict.cpu().clone().detach() + target.cpu().clone().detach(),2).float().sum()
    #     predict_1_and_False_unmber = predict_1_unmber - predict_1_and_True_unmber
    #     predict_0_and_True_unmber = torch.eq(predict.cpu().clone().detach() + target.cpu().clone().detach(),0).float().sum()
    #     predict_0_and_False_unmber = predict_0_unmber - predict_0_and_True_unmber
    #     message = f'Loss: {loss.item()}.'
    #     self.logger.info(message)                  
    #     message = f'predict_Acc: {(predict_1_and_True_unmber+predict_0_and_True_unmber)/predict.shape[0]}.'
    #     self.logger.info(message)  
    #     message = f'predict_IR: {predict_1_and_True_unmber/predict_1_unmber}.'
    #     self.logger.info(message)          

    # def classification_dataset_prepare(self, output_dict):
    #     one_data = {}
    #     ref_length_c = output_dict['ref_points_c'].shape[0]
    #     src_length_c = output_dict['src_points_c'].shape[0]
    #     gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps'].data
    #     gt_node_corr_indices = output_dict['gt_node_corr_indices'].data
    #     masks = torch.gt(gt_node_corr_overlaps, 0.0)
    #     gt_node_corr_indices = gt_node_corr_indices[masks]
    #     gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
    #     gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
    #     gt_node_corr_map = torch.zeros(ref_length_c, src_length_c)
    #     gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

    #     ref_node_corr_indices = output_dict['ref_node_corr_indices'].data
    #     src_node_corr_indices = output_dict['src_node_corr_indices'].data

    #     corr_node_ground_truth = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices]       

    #     # if self.mode != 'test':
    #     ground_truth_pos = torch.nonzero(gt_node_corr_map)
    #     ground_truth_neg = torch.nonzero(torch.eq(gt_node_corr_map,0))
    #     pos_index = [i for i in range(ground_truth_pos.shape[0])]
    #     random.shuffle(pos_index)
    #     ground_truth_pos = ground_truth_pos[pos_index[:2500],:]

    #     neg_index = [i for i in range(ground_truth_neg.shape[0])]
    #     random.shuffle(neg_index)
    #     ground_truth_neg = ground_truth_neg[neg_index[:ground_truth_pos.shape[0]],:]    

    #     ground_truth_both = torch.cat((ground_truth_pos,ground_truth_neg),dim=0)

    #     random_index = [i for i in range(ground_truth_both.shape[0])]
    #     random.shuffle(random_index)
    #     ground_truth_both = ground_truth_both[random_index]
    #     ref_node_corr_indices = ground_truth_both[:,0]
    #     src_node_corr_indices = ground_truth_both[:,1]
    #     corr_node_ground_truth = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices]

    #     ref_feats_c_norm = output_dict['ref_feats_c'].data
    #     src_feats_c_norm = output_dict['src_feats_c'].data

    #     ref_corr_node_feats = ref_feats_c_norm[ref_node_corr_indices]
    #     src_corr_node_feats = src_feats_c_norm[src_node_corr_indices]

    #     mean = src_corr_node_feats.mean()
    #     var = src_corr_node_feats.var()
    #     src_corr_node_feats = (src_corr_node_feats - mean) / torch.pow(var + 1e-05,0.5)  

    #     mean = ref_corr_node_feats.mean()
    #     var = ref_corr_node_feats.var()
    #     ref_corr_node_feats = (ref_corr_node_feats - mean) / torch.pow(var + 1e-05,0.5)  

    #     corr_node_feat = torch.cat((ref_corr_node_feats.unsqueeze(0).transpose(0,1), src_corr_node_feats.unsqueeze(0).transpose(0,1)), dim=1)
    #     corr_node_feat = corr_node_feat.repeat(1,1,2)
    #     corr_node_feat = corr_node_feat.chunk(16,dim=2)
    #     corr_node_feat = torch.cat((corr_node_feat),dim=1)  
    #     corr_node_feat = corr_node_feat.unsqueeze(1) 

    #     # one_data['file_name'] = (os.path.splitext('/'.join(file_path.split('/')[-3:])))[0]
    #     # one_data['src_corr_node_feats'] = src_corr_node_feats
    #     # one_data['ref_corr_node_feats'] = ref_corr_node_feats
    #     # one_data['src_corr_points'] = src_corr_points
    #     # one_data['ref_corr_points'] = ref_corr_points
    #     one_data['corr_node_feat'] = corr_node_feat
    #     one_data['ground_truth'] = corr_node_ground_truth

    #     return one_data


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
