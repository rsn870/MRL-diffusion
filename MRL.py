from typing import List

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import math

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float] = None, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        # relative importance shape: [G]
        self.relative_importance = relative_importance

    def forward(self, output, target):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]

        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)

        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()
    



class Radial_diffusion_Loss(nn.Module):
    """
    Compute geometry regularization objective using a multi-scale diffusive prior.
    """
    def __init__(self, relative_importance: List[float] = None, **kwargs):
        super(Radial_diffusion_Loss, self).__init__()
        self.criterion = nn.MSELoss(**kwargs)
        self.relative_importance = relative_importance
        self.beta_min = 0.0001
        self.beta_max = 0.02
        self.T = 125  # number of diffusion steps heuristic from Appendix B.10 cross fluc paper
        # Beta and alpha schedules
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.T)  # [T]
        self.alphas = 1.0 - self.betas  # [T]

    def forward(self, base_vector, nesting_vectors):
        # base_vector: [B, D]
        # nesting_vectors: list of [B, Ki]
        B, D = base_vector.shape
        device = base_vector.device

        # Permute batch for negative pairs
        perm = torch.randperm(B, device=device)
        base_shuffled = base_vector[perm].detach()

        losses = []

        # Squared L2 distances [B]
        base_sq_dist = ((base_vector - base_shuffled) ** 2).sum(dim=1)

        # Kernel shape: [B, T]
        alphas = self.alphas.to(device)
        betas = self.betas.to(device)
        base_kernel = torch.exp(-0.5 * (alphas / betas) * base_sq_dist.unsqueeze(1))

        for i, nest_vec in enumerate(nesting_vectors):
            nest_shuffled = nest_vec[perm].detach()
            B, K = nest_vec.shape

            nest_sq_dist = ((nest_vec - nest_shuffled) ** 2).sum(dim=1)
            # Upscale distance to adjust for dimensionality difference
            nest_sq_dist_up = (D / K) * nest_sq_dist

            nest_kernel = torch.exp(-0.5 * (alphas / betas) * nest_sq_dist_up.unsqueeze(1))

            loss_i = self.criterion(nest_kernel, base_kernel)
            losses.append(loss_i)

        losses = torch.stack(losses)  # [G]

        if self.relative_importance is None:
            rel_importance = torch.ones_like(losses, device=device)
        else:
            rel_importance = torch.tensor(self.relative_importance, device=device)

        weighted_loss = (rel_importance * losses).sum()
        return weighted_loss


class Angular_consistency_loss(nn.Module):
    """
    Compute an angular dot product prior across nested scales.
    """
    def __init__(self, relative_importance: List[float] = None, **kwargs):
        super(Angular_consistency_loss, self).__init__()
        self.criterion = nn.MSELoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(self, base_vector, nesting_vectors):
        # base_vector: [B, D]
        # nesting_vectors: list of [B, Ki]
        B, D = base_vector.shape
        device = base_vector.device

        # Permute batch for negative pairs
        perm = torch.randperm(B, device=device)
        base_shuffled = base_vector[perm].detach()

        # Normalize base vectors along dim=1 (per sample)
        base_norm = base_vector / base_vector.norm(dim=1, keepdim=True).clamp(min=1e-8)
        base_shuffled_norm = base_shuffled / base_shuffled.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Compute dot products batch-wise: [B]
        base_dot = (base_norm * base_shuffled_norm).sum(dim=1)

        losses = []

        for i, nest_vec in enumerate(nesting_vectors):
            nest_shuffled = nest_vec[perm].detach()
            B, K = nest_vec.shape

            nest_norm = nest_vec / nest_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            nest_shuffled_norm = nest_shuffled / nest_shuffled.norm(dim=1, keepdim=True).clamp(min=1e-8)

            nest_dot = (nest_norm * nest_shuffled_norm).sum(dim=1)

            # **Removed scaling (D/K) here as requested**

            # Compute loss per batch element and average
            loss_i = self.criterion(nest_dot, base_dot)
            losses.append(loss_i)

        losses = torch.stack(losses)  # [G]

        if self.relative_importance is None:
            rel_importance = torch.ones_like(losses, device=device)
        else:
            rel_importance = torch.tensor(self.relative_importance, device=device)

        weighted_loss = (rel_importance * losses).sum()
        return weighted_loss
    

class Combined_MRL_Loss(nn.Module):
    def __init__(
        self,
        ce_loss: Matryoshka_CE_Loss,
        radial_loss: Radial_diffusion_Loss,
        angular_loss: Angular_consistency_loss,
        radial_weight: float = 0.05,
        angular_weight: float = 0.05,
        warmup_steps: int = 10000,  # Number of steps before starting geometry losses
    ):
        super(Combined_MRL_Loss, self).__init__()
        self.ce_loss = ce_loss
        self.radial_loss = radial_loss
        self.angular_loss = angular_loss
        self.radial_weight = radial_weight
        self.angular_weight = angular_weight
        self.warmup_steps = warmup_steps
        self.current_step = 0  # To track progress

    def step(self):
        # Call this at every training step to increment
        self.current_step += 1

    def forward(self, outputs, targets, base_vector=None, nesting_vectors=None):
        # outputs, targets: for CE loss
        # base_vector, nesting_vectors: for diffusion and angular losses

        loss = self.ce_loss(outputs, targets)

        if self.current_step >= self.warmup_steps:
            if base_vector is not None and nesting_vectors is not None:
                loss += self.radial_weight * self.radial_loss(base_vector, nesting_vectors)
                loss += self.angular_weight * self.angular_loss(base_vector, nesting_vectors)

        return loss



class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes  # Number of classes for classification
        self.efficient = efficient
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x, geom_mode=False):
        nesting_logits = ()
        base_vector = x.clone()
        nesting_vectors = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()),)
                    nesting_vectors += (x[:, :num_feat].clone(),)
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias,)
                    nesting_vectors += (x[:, :num_feat].clone(),)
            else:
                nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
                nesting_vectors += (x[:, :num_feat].clone(),)
        if geom_mode:
            return nesting_logits, base_vector, nesting_vectors
        else:
            return nesting_logits


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following. 
    It effectively just look at the first "in_features" for the classification. 
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
