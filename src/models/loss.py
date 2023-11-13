import torch
import torch.nn as nn

from general_utils.tensor_utils import extract_non_diagonal_matrix
from models.FOCALModules import split_features


class FOCALLoss(nn.Module):
    def __init__(self, args):
        super(FOCALLoss, self).__init__()
        self.args = args
        self.config = args.dataset_config["FOCAL"]
        self.modalities = args.dataset_config["modality_names"]
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.orthonal_loss_f = nn.CosineEmbeddingLoss(reduction="mean")
        self.inter_ranking_loss_f = nn.MarginRankingLoss(margin=self.config["inter_rank_margin"], reduction="mean")

        # decide the temperature
        if isinstance(self.config["temperature"], dict):
            self.temperature = self.config["temperature"][args.model]
        else:
            self.temperature = self.config["temperature"]

    def mask_correlated_samples(self, seq_len, batch_size, temporal=False):
        """
        Return a mask where the positive sample locations are 0, negative sample locations are 1.
        """
        if temporal:
            """Extract comparison between sequences, output: [B * Seq, B * Seq]"""
            mask = torch.ones([batch_size, batch_size], dtype=bool).to(self.args.device)
            mask = mask.fill_diagonal_(0)
            mask = mask.repeat_interleave(seq_len, dim=0).repeat_interleave(seq_len, dim=1)
        else:
            """Extract [2N, 2N-2] negative locations from [2N, 2N] matrix, output: [seq, B, B]"""
            N = 2 * batch_size
            diag_mat = torch.eye(batch_size).to(self.args.device)
            mask = torch.ones((N, N)).to(self.args.device)

            mask = mask.fill_diagonal_(0)
            mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
            mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

            mask = mask.unsqueeze(0).repeat(seq_len, 1, 1).bool()

        return mask

    def forward_contrastive_loss(self, embeddings1, embeddings2, finegrain=False):
        """
        Among sequences, only samples at paired temporal locations are compared.
        embeddings shape: [b, seq, dim]
        """
        # get shape
        batch, seq, dim = embeddings1.shape

        # Put the compared dimension into the second dimension
        if finegrain:
            """Compare within the sequences, [b, seq, dim]"""
            in_embeddings1 = embeddings1
            in_embeddings2 = embeddings2
            N = 2 * seq
            dim_parallel = batch
            dim_compare = seq
        else:
            """Compare between the sequences, [seq, b, dim]"""
            in_embeddings1 = embeddings1.transpose(0, 1)
            in_embeddings2 = embeddings2.transpose(0, 1)
            N = 2 * batch
            dim_parallel = seq
            dim_compare = batch

        # Calculate similarity
        z = torch.cat((in_embeddings1, in_embeddings2), dim=1)
        sim = self.similarity_f(z.unsqueeze(2), z.unsqueeze(1)) / self.temperature
        sim_i_j = torch.diagonal(sim, dim_compare, dim1=-2, dim2=-1)
        sim_j_i = torch.diagonal(sim, -dim_compare, dim1=-2, dim2=-1)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=1).reshape(dim_parallel, N, 1)
        negative_samples = sim[self.mask_correlated_samples(dim_parallel, dim_compare)].reshape(dim_parallel, N, -1)

        # Compute loss
        labels = torch.zeros(dim_parallel * N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=2).reshape(dim_parallel * N, -1)
        contrastive_loss = self.criterion(logits, labels)

        return contrastive_loss

    def forward_orthogonality_loss(self, embeddings1, embeddings2):
        """
        Compute the orthogonality loss for the modality features. No cross-sample operation is involved.
        input shape: [b, seq_len, dim]
        We use y=-1 to make embedding1 and embedding2 orthogonal.
        """
        # convert [batch, seq, dim] to [batch * seq, dim]
        flat_embeddings1 = embeddings1.reshape(-1, embeddings1.shape[-1])
        flat_embeddings2 = embeddings2.reshape(-1, embeddings2.shape[-1])

        batch = flat_embeddings1.shape[0]
        orthogonal_loss = self.orthonal_loss_f(
            flat_embeddings1,
            flat_embeddings2,
            target=-torch.ones(batch).to(embeddings1.device),
        )

        return orthogonal_loss

    def forward_temporal_inter_ranking_loss(self, embeddings):
        """
        Enforce temporally close samples to be closer than temporally distant samples.
        embeddings: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = embeddings.shape
        in_embeddings = embeddings.reshape(batch_size * seq_len, dim)

        # calculate the Euclidean distance between any pairs of samples, [batch * seq, batch * seq]
        distance = torch.cdist(in_embeddings, in_embeddings, p=2)
        distance = distance.reshape(batch_size, seq_len, batch_size, seq_len)
        distance = distance.permute(0, 2, 1, 3)  # [b, b, seq, seq]

        # sequence level distance, [batch, batch]
        mask = torch.ones(batch_size * seq_len, batch_size * seq_len).to(self.args.device).fill_diagonal_(0)
        mask = mask.reshape(batch_size, seq_len, batch_size, seq_len).permute(0, 2, 1, 3)  # [b, b, seq, seq]
        distance = (distance * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])

        # in-sequence distance should be lower than inter-sequence distance
        avg_intra_seq_dist = torch.diagonal(distance, 0, dim1=0, dim2=1).repeat_interleave(batch_size - 1)
        avg_inter_seq_dist = extract_non_diagonal_matrix(distance).flatten()

        # target: x1 < x2, therefore y = -1
        ranking_loss = self.inter_ranking_loss_f(
            avg_intra_seq_dist,
            avg_inter_seq_dist,
            -torch.ones_like(avg_intra_seq_dist).to(self.args.device),
        )

        return ranking_loss

    def forward(self, mod_features1, mod_features2, index=None):
        """
        loss = shared contrastive loss + private contrastive loss + orthogonality loss + temporal correlation loss
        Procedure:
            (1) Split the features into (batch, subsequence, shared/private).
            (2) For each batch, compute the shared contrastive loss between modalities.
            (3) For each batch and modality, compute the private contrastive loss between samples.
            (4) Compute orthogonality loss beween shared-private and private-private representations.
            (5) For each subsequence, compute the temporal correlation loss.
        """
        seq_len = self.args.dataset_config["seq_len"]

        # Step 0: Reshape features
        reshaped_mod_features1, reshaped_mod_features2 = {}, {}
        for mod in self.modalities:
            reshaped_mod_features1[mod] = mod_features1[mod].reshape(-1, seq_len, mod_features1[mod].shape[-1])
            reshaped_mod_features2[mod] = mod_features2[mod].reshape(-1, seq_len, mod_features2[mod].shape[-1])

        # Step 1: split features into "shared" space and "private" space of each (mod, subsequence), [b, seq, dim]
        split_mod_features1 = split_features(reshaped_mod_features1)
        split_mod_features2 = split_features(reshaped_mod_features2)

        # Step 2: shared space contrastive loss
        shared_contrastive_loss = 0
        if self.args.tag == "noPrivate":
            for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            mod_features[mod1],
                            mod_features[mod2],
                        )
        else:
            for split_mod_features in [split_mod_features1, split_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            split_mod_features[mod1]["shared"],
                            split_mod_features[mod2]["shared"],
                        )

        # Step 3: private space contrastive loss
        private_contrastive_loss = 0
        for mod in self.modalities:
            private_contrastive_loss += self.forward_contrastive_loss(
                split_mod_features1[mod]["private"],
                split_mod_features2[mod]["private"],
            )

        # Step 4: temporal consistency loss
        temporal_consistency_loss = 0
        for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
            for mod in self.modalities:
                temporal_consistency_loss += self.forward_temporal_inter_ranking_loss(mod_features[mod])

        # Step 5: orthogonality loss
        orthogonality_loss = 0
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod in enumerate(self.modalities):
                # orthognoality between shared, private, and temporal space
                orthogonality_loss += self.forward_orthogonality_loss(
                    split_mod_features[mod]["shared"],
                    split_mod_features[mod]["private"],
                )

                # orthogonality between modalities
                for mod2 in self.modalities[i + 1 :]:
                    orthogonality_loss += self.forward_orthogonality_loss(
                        split_mod_features[mod]["private"],
                        split_mod_features[mod2]["private"],
                    )

        loss = (
            shared_contrastive_loss * self.config["shared_contrastive_loss_weight"]
            + private_contrastive_loss * self.config["private_contrastive_loss_weight"]
            + orthogonality_loss * self.config["orthogonal_loss_weight"]
            + temporal_consistency_loss * self.config["rank_loss_weight"]
        )

        return loss
