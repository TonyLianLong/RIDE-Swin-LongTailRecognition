import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Credit to: https://github.com/frank-xwang/RIDE-LongTailRecognition and https://github.com/kaidic/LDAM-DRW
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class LDAMWithSoftTargetCrossEntropy(LDAMLoss):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMWithSoftTargetCrossEntropy, self).__init__(cls_num_list=cls_num_list, max_m=max_m, weight=weight, s=s)

    def weighted_soft_cross_entropy(self, x, soft_target, hard_target, weight=None):
        loss = torch.sum(-soft_target * F.log_softmax(x, dim=-1), dim=-1)
        if weight is not None:
            loss = loss * weight[hard_target]
        return loss.mean()

    def forward(self, x, hard_target, soft_target=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, hard_target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return self.weighted_soft_cross_entropy(self.s*output, soft_target, weight=self.weight, hard_target=hard_target)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()

        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        # Perform adjustments on logits by adding margin (only enabled for LDAM)
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def base_loss(self, x, soft_target, hard_target, weight=None):
        # This is weighted soft cross-entropy since Swin-Transformer uses soft cross-entropy instead of the original cross-entropy.
        loss = torch.sum(-soft_target * F.log_softmax(x, dim=-1), dim=-1)
        if weight is not None:
            loss = loss * weight[hard_target]
        return loss.mean()


    def forward(self, output_logits, hard_target, soft_target=None, outputs_individual=None):
        loss = 0
        output_logits, outputs_individual = self.s * output_logits, self.s * outputs_individual

        if outputs_individual is None:
            return self.base_loss(output_logits, soft_target, hard_target, weight=self.per_cls_weights_base)

        # Construct loss with expert output, depending on the loss configuration
        for logits_item in outputs_individual:
            # Enable additional diversity factor on individual loss (does not make sense on collaborative loss)
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, soft_target, hard_target, weight=self.per_cls_weights_base)
            else:
                final_output = self.get_final_output(ride_loss_logits, hard_target)
                loss += self.base_loss_factor * self.base_loss(final_output, soft_target, hard_target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean') / len(outputs_individual)

        return loss

