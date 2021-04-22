import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
  def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma = None):
    super().__init__()

    self.samples_per_cls = samples_per_cls
    self.no_of_classes = no_of_classes
    self.loss_type = loss_type
    self.beta = beta
    self.gamma = gamma
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def forward(self, logits, labels):
    effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
    weights = (1.0 - self.beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * self.no_of_classes

    labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

    weights = torch.tensor(weights,device = self.device).float()
    weights = weights.unsqueeze(0).to(self.device)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,self.no_of_classes)

    if self.loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
    elif self.loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif self.loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss




