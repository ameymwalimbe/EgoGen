import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode

from prohmr.utils.geometry import rot6d_to_rotmat
from .fc_head_beta import FCHeadBeta


class SMPLFlowBeta(nn.Module):

    def __init__(self, cfg: CfgNode, contect_feats_dim=None, fc_head_beta_deep=False, fc_head_beta_condition_on_pose=False):
        """
        Probabilistic SMPL head using Normalizing Flows.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(SMPLFlowBeta, self).__init__()
        self.cfg = cfg
        self.npose = 6*(cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=contect_feats_dim)
        self.fc_head = FCHeadBeta(cfg, contect_feats_dim, fc_head_beta_deep, fc_head_beta_condition_on_pose)

    # Autocasting is disabled because SMPL has numerical instability issues with fp16 parameters.
    @autocast(enabled=False)
    def log_prob(self, smpl_params: Dict, feats: torch.Tensor) -> Tuple:
        """
        Compute the log-probability of a set of smpl_params given a batch of images.
        Args:
            smpl_params (Dict): Dictionary containing a set of SMPL parameters.
            feats (torch.Tensor): Conditioning features of shape (N, C).
        Returns:
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): The Gaussian latent corresponding to each sample with shape (B, N, 144).
        """

        feats = feats.float()
        batch_size = feats.shape[0]
        samples = torch.cat((smpl_params['global_orient'], smpl_params['body_pose']), dim=-1)  # [bs, 1, 144]
        num_samples = samples.shape[1]
        feats = feats.reshape(batch_size, 1, -1).repeat(1, num_samples, 1)  # [bs, 1, 2048/2048+x]
        # import pdb; pdb.set_trace()
        log_prob, z = self.flow.log_prob(samples.reshape(batch_size*num_samples, -1).to(feats.dtype), feats.reshape(batch_size*num_samples, -1))
        log_prob = log_prob.reshape(batch_size, num_samples)
        z = z.reshape(batch_size, num_samples, -1)
        return log_prob, z

    @autocast(enabled=False)
    def forward(self, feats: torch.Tensor, num_samples: Optional[int] = None, z: Optional[torch.Tensor] = None) -> Tuple:
        """
        Run a forward pass of the model.
        If z is not specified, then the model randomly draws num_samples samples for each image in the batch.
        Otherwise the batch of latent vectors z is transformed using the Conditional Normalizing Flows model.
        Args:
            feats (torch.Tensor): Conditioning features of shape (N, C).
            num_samples (int): Number of samples to draw per image.
            z (torch.Tensor): A batch of latent vectors of shape (B, N, 144).
        Returns:
            pred_smpl_params (Dict): Dictionary containing the predicted set of SMPL parameters.
            pred_cam (torch.Tensor): Predicted camera parameters with shape (B, N, 3).
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): Either the input z or the randomly drawn batch of latent Gaussian vectors.
            pred_pose_6d (torch.Tensor): Predicted pose vectors in the 6-dimensional representation.
        """

        feats = feats.float()

        batch_size = feats.shape[0]

        if z is None:
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats)
            z = z.reshape(batch_size, num_samples, -1)
            pred_params = samples.reshape(batch_size, num_samples, -1)
        else:
            num_samples = z.shape[1]
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats, noise=z)
            pred_params = samples.reshape(batch_size, num_samples, -1)


        pred_pose = pred_params[:, :, :self.npose]  # [bs, 1/num_sample, 144]  self.npose=144
        pred_pose_6d = pred_pose.clone()
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose.reshape(batch_size * num_samples, -1)).view(batch_size, num_samples, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)  # [bs, 1/num_sample, 24, 3, 3]
        pred_smpl_params = {'global_orient': pred_pose_rotmat[:, :, [0]],
                             'body_pose': pred_pose_rotmat[:, :, 1:]}
        pred_betas = self.fc_head(pred_smpl_params, pred_pose, feats)  # [bs, num_samples, 10]
        pred_smpl_params['betas'] = pred_betas

        return pred_smpl_params, log_prob, z, pred_pose_6d
