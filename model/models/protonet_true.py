import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.base_true import FewShotModel


def compute_emb_cosine_similarity(support_emb: torch.Tensor, query_emb: torch.Tensor):
    """Compute the similarity matrix C between support and query embeddings using the cosine similarity.
       We reformulate the cosine sim computation as dot product using matrix mult and transpose, due to:
       cossim = dot(u,v)/(u.norm * v.norm) = dot(u/u.norm, v/v.norm);    u/u.norm = u_norm, v/v.norm = v_norm
       For two matrices (tensors) U and V of shapes [n,b] & [m,b] => torch.mm(U_norm,V_norm.transpose)
       This returns a matrix showing all pairwise cosine similarities of all entries in both matrices, i.e. [n,m]"""

    # Note: support represents the 'reference' embeddings, query represents the ones that need classification;
    #       During adaptation of peiv, support set embeddings will be used for both to optimise the peiv
    support_shape = support_emb.shape
    support_emb_vect = support_emb.reshape(support_shape[0] * support_shape[1], -1)  # shape e.g. [4900, 384]
    # Robust version to avoid division by zero
    support_norm = torch.linalg.norm(support_emb_vect, dim=1).unsqueeze(dim=1)
    support_norm = support_emb_vect / torch.max(support_norm, torch.ones_like(support_norm) * 1e-8)

    query_shape = query_emb.shape
    query_emb_vect = query_emb.reshape(query_shape[0] * query_shape[1], -1)  # shape e.g. [14700, 384]
    # Robust version to avoid division by zero
    query_norm = query_emb_vect.norm(dim=1).unsqueeze(dim=1)
    query_norm = query_emb_vect / torch.max(query_norm, torch.ones_like(query_norm) * 1e-8)

    return torch.matmul(support_norm, query_norm.transpose(0, 1))  # shape e.g. [4900, 14700]


class ProtoNetTrue(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.total_len_support_key = args.way * args.shot * args.seq_len
        # Mask to prevent image self-classification during adaptation
        if args.shot > 1:  # E.g. for 5-shot scenarios, use 'full' block-diagonal logit matrix to mask entire image
            self.block_mask = torch.block_diag(*[torch.ones(args.seq_len, args.seq_len) * -100.
                                                 for _ in range(args.way * args.shot)]).cuda()
        else:  # 1-shot experiments require diagonal in-image masking, since no other samples available
            self.block_mask = torch.ones(args.seq_len * args.way * args.shot,
                                         args.seq_len * args.way * args.shot).cuda()
            self.block_mask = (self.block_mask - self.block_mask.triu(diagonal=args.block_mask_1shot)
                               - self.block_mask.tril(diagonal=-args.block_mask_1shot)) * -1000.

        self.v = torch.zeros(self.total_len_support_key, requires_grad=True, device='cuda')

        self.log_tau_c = torch.tensor([np.log(args.similarity_temp_init)], requires_grad=True, device='cuda')

        self.peiv_init_state = True
        self.disable_peiv_optimisation = args.disable_peiv_optimisation
        self.optimiser_str = args.optimiser_online
        self.opt_steps = args.optim_steps_online
        self.lr_online = args.lr_online

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _reset_peiv(self):
        """Reset the patch embedding importance vector to zeros"""
        # Re-create patch importance vector (and add to optimiser in _optimise_peiv() -- might exist a better option)
        self.v = torch.zeros(self.total_len_support_key, requires_grad=True, device="cuda")

    def _predict(self, support_emb, query_emb, phase='infer'):
        """Perform one forward pass using the provided embeddings as well as the module-internal
        patch embedding importance vector 'peiv'. The phase parameter denotes whether the prediction is intended for
        adapting peiv ('adapt') using the support set, or inference ('infer') on the query set."""
        sup_emb_seq_len = support_emb.shape[1]
        # Compute patch embedding similarity
        pred = compute_emb_cosine_similarity(support_emb, query_emb)
        # Mask out block diagonal during adaptation to prevent image patches from classifying themselves and neighbours
        # if phase == 'adapt':
        #     C = C + self.block_mask
        # Add patch embedding importance vector (logits, corresponds to multiplication of probabilities)
        # pred = torch.add(C, self.v.unsqueeze(1))  # using broadcasting
        # =========
        # Rearrange the patch dimensions to combine the embeddings
        pred = pred.view(self.args.way, self.args.shot * sup_emb_seq_len,
                         query_emb.shape[0], query_emb.shape[1]).transpose(2, 3)
        # Reshape to combine all embeddings related to one query image
        pred = pred.reshape(self.args.way, self.args.shot * sup_emb_seq_len * query_emb.shape[1], query_emb.shape[0])
        # Temperature scaling
        pred = pred / torch.exp(self.log_tau_c)
        # Gather log probs of all patches for each image
        pred = torch.logsumexp(pred, dim=1)
        # Return the predicted logits
        return pred.transpose(0, 1)

    def _optimise_peiv(self, support_emb_key, support_emb_query, supp_labels):
        # Detach, we don't want to compute computational graph w.r.t. model
        support_emb_key = support_emb_key.detach()
        support_emb_query = support_emb_query.detach()
        supp_labels = supp_labels.detach()
        params_to_optimise = [self.v]
        # Perform optimisation of patch embedding importance vector v; embeddings should be detached here!
        self.optimiser_online = torch.optim.SGD(params=params_to_optimise, lr=self.lr_online)
        self.optimiser_online.zero_grad()
        # Run for a specific number of steps 'self.opt_steps' using SGD
        for s in range(self.opt_steps):
            # 允许在with no grad上下文中使用梯度计算
            with torch.enable_grad():
                support_pred = self._predict(support_emb_key, support_emb_query, phase='adapt')
                loss = self.loss_fn(support_pred, supp_labels)
            loss.backward()
            self.optimiser_online.step()
            self.optimiser_online.zero_grad()
        # Set initialisation/reset flag to False since peiv is no longer 'just' initialised
        self.peiv_init_state = False
        return

    def _forward(self, support_emb_key, support_emb_query, query_emb, support_labels):
        # Check whether patch importance vector has been reset to its initialisation state
        # if not self.peiv_init_state:
        #     self._reset_peiv()
        # Run optimisation on peiv
        # if not self.disable_peiv_optimisation:
        #     self._optimise_peiv(support_emb_key, support_emb_query, support_labels)
        # Retrieve the predictions of query set samples
        pred_query = self._predict(support_emb_key, query_emb, phase='infer')
        return pred_query