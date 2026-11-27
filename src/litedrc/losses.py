import torch
import torch.nn as nn

class BetaEDLLoss(nn.Module):
    def __init__(self, annealing_epochs=50):
        super().__init__()
        self.annealing_epochs = annealing_epochs

    def forward(self, alpha, beta, targets, current_epoch):
        S = alpha + beta
        probs = alpha / S
        
        loss_err = (targets - probs) ** 2 + (alpha * beta) / (S ** 2 * (S + 1.0))
        
        alpha_tilde = targets + (1 - targets) * alpha
        beta_tilde = (1 - targets) + targets * beta
        S_tilde = alpha_tilde + beta_tilde
        
        kl_loss = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) \
                  + (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)) \
                  + (beta_tilde - 1) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
                  
        annealing_coef = min(1.0, current_epoch / self.annealing_epochs)
        loss = loss_err + annealing_coef * kl_loss
        return loss.sum(dim=1).mean()
