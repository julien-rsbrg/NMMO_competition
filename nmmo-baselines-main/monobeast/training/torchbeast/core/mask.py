import torch
import torch.nn.functional as F


class MaskedPolicy:
    def __init__(self, logits, valid_actions=None):
        self.valid_actions = valid_actions
        if valid_actions is None:
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        else:
            logits = valid_actions * logits + (1 - valid_actions) * (-1e8)
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)

    def sample(self):
        u = torch.rand_like(self.logits, device=self.logits.device)
        return torch.argmax(self.logits - torch.log(-torch.log(u)), axis=-1)

    def argmax(self):
        return torch.argmax(self.logits, axis=-1)

    def log_prob(self, action):
        action = action.long().unsqueeze(-1)
        return torch.gather(self.logits, -1, action)

    def entropy(self):
        policy = F.softmax(self.logits, dim=-1)
        log_policy = F.log_softmax(self.logits, dim=-1)
        return policy * log_policy
