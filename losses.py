from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR

    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # mask = mask * logits_mask

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SINCERELoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.Tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """

        device = embeds.device

        # Garantir que embeddings estejam em float16
        embeds = embeds.to(dtype=torch.float16, device=device)
        labels = labels.to(dtype=torch.int64, device=device)  # Labels devem ser inteiros

        # Calcular logits (B, B)
        logits = torch.matmul(embeds, embeds.T).to(dtype=torch.float16)
        logits /= self.temperature

        # Determinar pares de mesmo rótulo
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Criar máscara para o denominador
        denom_activations = torch.full_like(logits, -65504.0, dtype=torch.float16)  # Evita erro com -inf
        denom_activations[~same_label] = logits[~same_label]

        # Calcular logsumexp para o denominador
        base_denom_row = torch.logsumexp(denom_activations, dim=0).to(dtype=torch.float16)
        base_denom = base_denom_row.unsqueeze(1).repeat((1, base_denom_row.shape[0]))

        # Criar máscara para o numerador, removendo comparações consigo mesmo
        in_numer = same_label.clone()
        in_numer.fill_diagonal_(False)  # Método seguro para evitar auto-comparação

        # Contar elementos no numerador
        numer_count = in_numer.sum(dim=0).to(dtype=torch.float16)

        # Criar matriz de logits do numerador
        numer_logits = torch.zeros_like(logits, dtype=torch.float16)
        numer_logits[in_numer] = logits[in_numer]

        # Criar denominador logsumexp de forma segura
        stacked = torch.stack([numer_logits[in_numer], base_denom[in_numer]], dim=0).to(dtype=torch.float16)
        log_denom = torch.zeros_like(logits, dtype=torch.float16)
        log_denom[in_numer] = torch.logsumexp(stacked, dim=0).to(dtype=torch.float16)

        # Calcular entropia cruzada
        ce = -1 * (numer_logits - log_denom)

        # Normalizar e calcular a perda
        loss = torch.sum(ce / numer_count) / ce.shape[0]

        return loss


# class SINCERELoss(nn.Module):
#     def __init__(self, temperature=0.07) -> None:
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, embeds: torch.Tensor, labels: torch.tensor):
#         """Supervised InfoNCE REvisited loss with cosine distance

#         Args:
#             embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
#             labels (torch.tensor): (B,) integer class labels.

#         Returns:
#             torch.Tensor: Scalar loss.
#         """

#         device = (embeds.get_device()
#                   if embeds.is_cuda
#                   else torch.device('cpu'))

#         # calculate logits (activations) for each embeddings pair (B, B)
#         # using matrix multiply instead of cosine distance function for ~10x cost reduction
#         embeds = embeds.to(dtype=torch.float16, device=device)
#         logits = embeds @ embeds.T
#         logits = logits.to(dtype=torch.float16)
#         # logits = embeds @ embeds.T
#         # logits /= self.temperature
#         # determine which logits are between embeds of the same label (B, B)
#         same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

#         # masking with -inf to get zeros in the summation for the softmax denominator
#         denom_activations = torch.full_like(logits, float("-inf"), dtype=torch.float16)
#         denom_activations[~same_label] = logits[~same_label]
#         # get logsumexp of the logits between embeds of different labels for each row (B,)
#         base_denom_row = torch.logsumexp(denom_activations, dim=0)
#         # reshape to be (B, B) with row values equivalent, to be masked later
#         base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

#         # get mask for numerator terms by removing comparisons between an image and itself (B, B)
#         in_numer = same_label
#         in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
#         # delete same_label so don't need to copy for in_numer
#         del same_label
#         # count numerator terms for averaging (B,)
#         numer_count = in_numer.sum(dim=0).to(dtype=torch.float16, device=device)
#         # numerator activations with others zeroed (B, B)
#         numer_logits = torch.zeros_like(logits, dtype=torch.float16)
#         numer_logits[in_numer] = logits[in_numer]

#         # construct denominator term for each numerator via logsumexp over a stack (B, B)
#         log_denom = torch.zeros_like(logits, dtype=torch.float16)
#         log_denom[in_numer] = torch.stack(
#             (numer_logits[in_numer], base_denom[in_numer]), dim=0
#         ).logsumexp(dim=0)

#         # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
#         # entries not in numerator set to 0
#         ce = -1 * (numer_logits - log_denom)
#         # take average over rows with entry count then average over batch
#         loss = torch.sum(ce / numer_count.to(device)) / ce.shape[0]

#         return loss


class MultiviewSINCERELoss(SINCERELoss):
    def __init__(self, temperature=0.07) -> None:
        super().__init__(temperature)

    def forward(self, embeds, labels: torch.tensor):
        """Supervised InfoNCE with cosine distance and multiple image views

        Args:
            embeds (torch.Tensor): (B, V, D) embeddings of V augmented views of B images,
                                   normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # collapse view dimension, leaving views next to each other
        reshaped_embeds = torch.reshape(embeds, (-1, embeds.shape[2]))
        # repeat labels to account for views
        labels = labels.repeat_interleave(embeds.shape[1])

        return super().forward(reshaped_embeds, labels)


class InfoNCELoss(SINCERELoss):
    def __init__(self, temperature=0.07) -> None:
        super().__init__(temperature)

    def forward(self, embeds):
        """InfoNCE with cosine distance

        Args:
            embeds (torch.Tensor): (B, V, D) embeddings of V augmented views of B images,
                                   normalized over D dimension.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # collapse view dimension, leaving views next to each other
        reshaped_embeds = torch.reshape(embeds, (-1, embeds.shape[2]))
        # create self-supervised labels
        labels = torch.arange(0, embeds.shape[0]).repeat_interleave(embeds.shape[1])
        return super().forward(reshaped_embeds, labels)

class EpsSupInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, epsilon=0.25) -> None:
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.Tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        device = embeds.device

        # Garantir que os tensores estejam em float16
        embeds = embeds.to(dtype=torch.float16, device=device)
        labels = labels.to(dtype=torch.int64, device=device)

        # Calcular logits (B, B)
        logits = torch.matmul(embeds, embeds.T).to(dtype=torch.float16)
        logits /= self.temperature

        # Determinar pares de mesmo rótulo
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Criar máscara para o denominador
        denom_activations = torch.full_like(logits, -65504.0, dtype=torch.float16)  # Evita erro com -inf
        denom_activations[~same_label] = logits[~same_label]

        # Calcular logsumexp para o denominador
        base_denom_row = torch.logsumexp(denom_activations, dim=0).to(dtype=torch.float16)
        base_denom = base_denom_row.unsqueeze(1).repeat((1, base_denom_row.shape[0]))

        # Criar máscara para o numerador, removendo comparações consigo mesmo
        in_numer = same_label.clone()
        in_numer.fill_diagonal_(False)  # Método seguro para evitar auto-comparação

        # Contar elementos no numerador
        numer_count = in_numer.sum(dim=0).to(dtype=torch.float16)

        # Criar matriz de logits do numerador
        numer_logits = torch.zeros_like(logits, dtype=torch.float16)
        numer_logits[in_numer] = logits[in_numer]

        # Criar denominador logsumexp de forma segura
        stacked = torch.stack([
            (numer_logits[in_numer] - self.epsilon).to(dtype=torch.float16),  # Corrigido
            base_denom[in_numer]
        ], dim=0).to(dtype=torch.float16)  # Garantia de float16

        log_denom = torch.zeros_like(logits, dtype=torch.float16)
        log_denom[in_numer] = torch.logsumexp(stacked, dim=0).to(dtype=torch.float16)

        # Calcular entropia cruzada
        ce = -1 * (numer_logits - log_denom)

        # Normalizar e calcular a perda
        loss = torch.sum(ce / numer_count) / ce.shape[0]

        return loss
   
# class EpsSupInfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07, epsilon=0.25) -> None:
#         super().__init__()
#         self.temperature = temperature
#         self.epsilon = epsilon

#     def forward(self, embeds: torch.Tensor, labels: torch.tensor):
#         """Supervised InfoNCE REvisited loss with cosine distance

#         Args:
#             embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
#             labels (torch.tensor): (B,) integer class labels.

#         Returns:
#             torch.Tensor: Scalar loss.
#         """
#         # calculate logits (activations) for each embeddings pair (B, B)
#         # using matrix multiply instead of cosine distance function for ~10x cost reduction
#         logits = embeds @ embeds.T
#         logits /= self.temperature
#         # determine which logits are between embeds of the same label (B, B)
#         same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

#         # masking with -inf to get zeros in the summation for the softmax denominator
#         denom_activations = torch.full_like(logits, float("-inf"))
#         denom_activations[~same_label] = logits[~same_label]
#         # get logsumexp of the logits between embeds of different labels for each row (B,)
#         base_denom_row = torch.logsumexp(denom_activations, dim=0)
#         # reshape to be (B, B) with row values equivalent, to be masked later
#         base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

#         # get mask for numerator terms by removing comparisons between an image and itself (B, B)
#         in_numer = same_label
#         in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
#         # delete same_label so don't need to copy for in_numer
#         del same_label
#         # count numerator terms for averaging (B,)
#         numer_count = in_numer.sum(dim=0)
#         # numerator activations with others zeroed (B, B)
#         numer_logits = torch.zeros_like(logits)
#         numer_logits[in_numer] = logits[in_numer]

#         # construct denominator term for each numerator via logsumexp over a stack (B, B)
#         log_denom = torch.zeros_like(logits)
#         log_denom[in_numer] = torch.stack(
#             (numer_logits[in_numer] - self.epsilon, base_denom[in_numer]), dim=0).logsumexp(dim=0)
#         # note that the subtraction of epsilon on previous line is only difference from SINCERE

#         # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
#         # entries not in numerator set to 0
#         ce = -1 * (numer_logits - log_denom)
#         # take average over rows with entry count then average over batch
#         loss = torch.sum(ce / numer_count) / ce.shape[0]

#         return loss


class MultiviewEpsSupInfoNCELoss(EpsSupInfoNCELoss):
    def __init__(self, temperature=0.07, epsilon=0.25) -> None:
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, embeds, labels: torch.tensor):
        """Supervised InfoNCE with cosine distance and multiple image views

        Args:
            embeds (torch.Tensor): (B, V, D) embeddings of V augmented views of B images,
                                   normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # collapse view dimension, leaving views next to each other
        reshaped_embeds = torch.reshape(embeds, (-1, embeds.shape[2]))
        # repeat labels to account for views
        labels = labels.repeat_interleave(embeds.shape[1])

        return super().forward(reshaped_embeds, labels)
