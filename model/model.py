import time
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.core import Registry as R
from torchdrug.data import PackedProtein
from torchdrug.layers import functional


@R.register("models.FusionNetwork")
class FusionNetwork(nn.Module, core.Configurable):
    def __init__(self, sequence_model, structure_model, fusion="series", cross_dim=None):
        super(FusionNetwork, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.output_dim = sequence_model.output_dim + structure_model.output_dim
        self.inject_step = 5   # (sequence_layers / structure_layers) layers

        # Structure embeddings layer
        raw_input_dim = 21  # amino acid tokens
        self.structure_embed_linear = nn.Linear(raw_input_dim, structure_model.input_dim)
        self.embedding_batch_norm = nn.BatchNorm1d(structure_model.input_dim)

        # Normal Initialization of the 3D structure token
        structure_token = nn.Parameter(torch.Tensor(structure_model.input_dim).unsqueeze(0))
        nn.init.normal_(structure_token, mean=0.0, std=0.01)
        self.structure_token = nn.Parameter(structure_token.squeeze(0))

        # Linear Transformation between structure to sequential spaces
        self.structure_linears = nn.ModuleList([nn.Linear(structure_model.dims[-1], sequence_model.output_dim)
                                                for _ in range(self.inject_step + 1)])
        self.seq_linears = nn.ModuleList([nn.Linear(sequence_model.output_dim, structure_model.dims[-1])
                                         for _ in range(self.inject_step + 1)])

    def forward(self, graph, input, all_loss=None, metric=None):
        # Build a new protein graph with the 3D token (the lase node)
        new_graph = self.build_protein_graph_with_3d_token(graph)

        # Sequence (ESM) model initialization
        sequence_input = self.sequence_model.mapping[graph.residue_type]
        sequence_input[sequence_input == -1] = graph.residue_type[sequence_input == -1]
        size = graph.num_residues

        # Check if sequence size is not bigger than max seq length
        if (size > self.sequence_model.max_input_length).any():
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.sequence_model.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residues)
            sequence_input = sequence_input[mask]
            graph = graph.subresidue(mask)
        size_ext = size

        # BOS == CLS
        if self.sequence_model.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.sequence_model.device) * self.sequence_model.alphabet.cls_idx
            sequence_input, size_ext = functional._extend(bos, torch.ones_like(size_ext), sequence_input, size_ext)

        if self.sequence_model.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.sequence_model.device) * self.sequence_model.alphabet.eos_idx
            sequence_input, size_ext = functional._extend(sequence_input, size_ext, eos, torch.ones_like(size_ext))

        # Padding
        tokens = functional.variadic_to_padded(sequence_input, size_ext, value=self.sequence_model.alphabet.padding_idx)[0]
        repr_layers = [self.sequence_model.repr_layer]
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.sequence_model.model.padding_idx)  # B, T

        # Sequence embedding layer
        x = self.sequence_model.model.embed_scale * self.sequence_model.model.embed_tokens(tokens)

        if self.sequence_model.model.token_dropout:
            x.masked_fill_((tokens == self.sequence_model.model.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.sequence_model.model.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # Structure model initialization
        structure_hiddens = []
        batch_size = graph.batch_size
        structure_embedding = self.embedding_batch_norm(self.structure_embed_linear(input))
        structure_token_batched = self.structure_token.unsqueeze(0).expand(batch_size, -1)
        structure_input = torch.cat([structure_embedding.squeeze(1), structure_token_batched], dim=0)

        # Add the 3D token representation
        structure_token_expanded = self.structure_token.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1)  # (B, 1, E)
        x = torch.cat((x[:, :-1], structure_token_expanded, x[:, -1:]), dim=1)  # (B, T + 1, E)
        padding_mask = torch.cat([padding_mask[:, :-1],
                                  torch.zeros(padding_mask.size(0), 1).to(padding_mask), padding_mask[:, -1:]], dim=1)
        size_ext += 1

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None

        # Layers forward pass
        structure_layer_index = 0
        for seq_layer_idx, seq_layer in enumerate(self.sequence_model.model.layers):
            x, attn = seq_layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )
            if (seq_layer_idx + 1) in repr_layers:
                hidden_representations[seq_layer_idx + 1] = x.transpose(0, 1)

            # Inject structure knowledge every inject_step (sequence_layers / structure_layers) layers
            if seq_layer_idx > 0 and seq_layer_idx % self.inject_step == 0:
                if structure_layer_index == 0:
                    structure_input = torch.cat((structure_input[:-1 * batch_size],  x[-2, :, :]), dim=0)
                else:
                    structure_input = torch.cat((structure_input[:-1 * batch_size],
                                                 self.seq_linears[structure_layer_index](x[-2, :, :])), dim=0)

                hidden = self.structure_model.layers[structure_layer_index](new_graph, structure_input)
                if self.structure_model.short_cut and hidden.shape == structure_input.shape:
                    hidden = hidden + structure_input
                if self.structure_model.batch_norm:
                    hidden = self.structure_model.batch_norms[structure_layer_index](hidden)

                structure_hiddens.append(hidden)
                structure_input = hidden

                # Update the 3D token
                updated_structure_token = self.structure_linears[structure_layer_index](structure_input[-1 * batch_size:])
                x = torch.cat((x[:-2, :, :], updated_structure_token.unsqueeze(0), x[-1:, :, :]), dim=0)
                structure_layer_index += 1

        # Structural Output
        if self.structure_model.concat_hidden:
            structure_node_feature = torch.cat(structure_hiddens, dim=-1)[:-1 * batch_size]
        else:
            structure_node_feature = structure_hiddens[-1][:-1 * batch_size]

        structure_graph_feature = self.structure_model.readout(graph, structure_node_feature)

        # Sequence Output
        x = self.sequence_model.model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (seq_layer_idx + 1) in repr_layers:
            hidden_representations[seq_layer_idx + 1] = x
        x = self.sequence_model.model.lm_head(x)

        output = {"logits": x, "representations": hidden_representations}

        # Sequence (ESM) model outputs
        residue_feature = output["representations"][self.sequence_model.repr_layer]
        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.sequence_model.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.sequence_model.readout(graph, residue_feature)

        # Combine both models outputs
        node_feature = torch.cat([residue_feature, structure_node_feature], dim=-1)
        graph_feature = torch.cat([graph_feature, structure_graph_feature], dim=-1)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

    @staticmethod
    def build_protein_graph_with_3d_token(graph):
        batch_size = graph.num_nodes.size(0)
        num_nodes = graph.num_nodes
        max_num_nodes = num_nodes.max().item()
        total_nodes = num_nodes.sum().item()

        # Create new edges for each graph in the batch
        node_indices = torch.arange(max_num_nodes, dtype=torch.int64, device=graph.edge_list.device)
        node_indices = node_indices.unsqueeze(0).expand(batch_size, -1)

        # Unique 3D token index for each graph, placed at the end
        new_bond_type_value = graph.bond_type.max().item() + 1

        new_edges_from_token = torch.stack([
            torch.arange(batch_size, device=graph.edge_list.device).view(-1, 1).repeat(1, max_num_nodes),
            node_indices,
            torch.full_like(node_indices, new_bond_type_value, dtype=torch.int64, device=graph.edge_list.device)
        ], dim=-1)

        new_edges_to_token = torch.stack([
            node_indices,
            torch.arange(batch_size, device=graph.edge_list.device).view(-1, 1).repeat(1, max_num_nodes),
            torch.full_like(node_indices, new_bond_type_value, dtype=torch.int64, device=graph.edge_list.device)
        ], dim=-1)

        # Correctly set the first column of new_edges_from_token to total_nodes + batch index
        new_edges_from_token[:, :, 0] = total_nodes + torch.arange(batch_size, dtype=torch.int64,
                                                                   device=graph.edge_list.device).unsqueeze(1)

        # Correctly set the second column of new_edges_to_token to total_nodes + batch index
        new_edges_to_token[:, :, 1] = total_nodes + torch.arange(batch_size, dtype=torch.int64,
                                                                 device=graph.edge_list.device).unsqueeze(1)

        mask = (node_indices < num_nodes.unsqueeze(1))
        new_edges_from_token = new_edges_from_token[mask].view(-1, 3)
        new_edges_to_token = new_edges_to_token[mask].view(-1, 3)

        # Correct the middle column to properly index nodes within each graph for new_edges_from_token
        batch_offsets = torch.cat([torch.tensor([0], device=graph.edge_list.device), num_nodes.cumsum(0)[:-1]])
        new_edges_from_token[:, 1] += batch_offsets.repeat_interleave(num_nodes)

        # Correct the first column to properly index nodes within each graph for new_edges_to_token
        new_edges_to_token[:, 0] += batch_offsets.repeat_interleave(num_nodes)

        # Concatenate the original edges with the new edges
        all_edges = torch.cat([graph.edge_list, new_edges_from_token, new_edges_to_token], dim=0)

        # Create new atom type including the 3D token
        new_atom_type = torch.cat([
            graph.atom_type,
            torch.full((batch_size,), graph.atom_type[-1].item(), dtype=torch.int64, device=graph.atom_type.device)
        ], dim=0)

        new_bond_type = torch.cat([
            graph.bond_type,
            torch.full((new_edges_from_token.size(0) + new_edges_to_token.size(0),), new_bond_type_value,
                       dtype=torch.int64, device=graph.bond_type.device)
        ], dim=0)

        new_num_nodes = graph.num_nodes + 1
        new_num_edges = graph.num_edges + 2 * graph.num_nodes
        new_num_residues = graph.num_residues + 1
        new_num_relation = all_edges[:, 2].max() + 1

        offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=num_nodes.device),
                             num_nodes.cumsum(dim=0)])[:-1].repeat_interleave(new_num_edges)

        new_graph = PackedProtein(edge_list=all_edges, atom_type=new_atom_type, bond_type=new_bond_type,
                                  residue_type=graph.residue_type, view=graph.view, num_nodes=new_num_nodes,
                                  num_edges=new_num_edges, num_residues=new_num_residues, offsets=offsets,
                                  num_relation=new_num_relation)
        return new_graph
