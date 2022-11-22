from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from torch_geometric.utils import to_networkx

mol_feat_dict = {
    # The node features are: Atom Symbol (13), Degree (6), Formal Charge (1), Radical Electrons (1),
    # Hybridization (7), Aromaticity (1), Hydrogens (5), Chirality (1), Chirality Type (2). Total: 37
    "Atom Symbol": (0, 12),
    "Degree": (12, 18),
    "Formal Charge": (18, 19),
    "Radical Electrons": (19, 20),
    "Hybridization": (21, 28),
    "Aromaticity": (28, 29),
    "Hydrogens": (29, 34),
    "Chirality": (34, 35),
    "Chirality Type": (35, 37),
    # The Edge features are: Bond Type (4), Conjugation (1), Ring (1), Stereo (4). Total: 10.
    "Bond Type": (0, 4),
    "Conjugation": (4, 5),
    "Ring": (5, 6),
    "Stereo": (6, 10)
}


def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
    plt.show()


def to_molecule(data):
    # ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F',
    #             'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    ATOM_MAP = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'Pt', 'S', 'other']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = ATOM_MAP[data['x'].index(1.0)]
        del data['x']
    return g


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += float(val.sum())

    # min-max scale to 0-1 range, want to use attribution as alpha while highlighting the molecule
    cur_min = min(edge_mask_dict.values())
    cur_max = max(edge_mask_dict.values())
    for key in edge_mask_dict.keys():
        if edge_mask_dict[key] == 0.0:
            continue

        edge_mask_dict[key] = (edge_mask_dict[key] - cur_min) / (cur_max - cur_min)

    return edge_mask_dict


def aggregate_node(node_mask):
    node_mask_dict = defaultdict(float)
    for row_i in list(range(node_mask.shape[0])):
        node_mask_dict[row_i] = float(node_mask[row_i, ].sum())

    cur_min = min(node_mask_dict.values())
    cur_max = max(node_mask_dict.values())
    for key in node_mask_dict.keys():
        if node_mask_dict[key] == 0.0:
            continue
        else:
            node_mask_dict[key] = (node_mask_dict[key] - cur_min) / (cur_max - cur_min)

    return node_mask_dict


def mask_molecules(cur_attr, lower_idx, upper_idx):
    mask = torch.ones(size=cur_attr.shape, dtype=bool)
    mask[:, lower_idx:upper_idx] = False
    final_attr = cur_attr.masked_fill_(mask, 0.0)

    return final_attr


def drug_interpret_viz(edge_attr, node_attr, drug_graph, sample_info, plot_address, target, aggregate: bool = True):
    print("Starting drug plotting...")
    # Separate positive and negative attributions by clamping them
    neg_node_mask = node_attr.clamp(max=0)
    neg_edge_mask = edge_attr.clamp(max=0)
    pos_node_mask = node_attr.clamp(min=0)
    pos_edge_mask = edge_attr.clamp(min=0)

    # sum up across different node and edge feature attributions
    # neg_edge_mask = neg_edge_mask.sum(axis=1)
    # pos_edge_mask = pos_edge_mask.sum(axis=1)
    # neg_node_mask = neg_node_mask.sum(axis=1)
    # pos_node_mask = pos_node_mask.sum(axis=1)

    rdkit_mol = Chem.MolFromSmiles(sample_info['smiles'][0])
    legend_pos = "Compound: " + sample_info['drug_name'][0] + ", Cell Line: " + sample_info['cell_line_name'][
        0] + ", AAC: " + target + " Positive Attributions"
    legend_neg = "Compound: " + sample_info['drug_name'][0] + ", Cell Line: " + sample_info['cell_line_name'][
        0] + ", AAC: " + target + ", Negative Attributions"

    blank_edge_colors = {i: (1, 0, 0, 0) for i in range(len(edge_attr))}
    blank_edge_highlights = list(range(len(blank_edge_colors)))

    blank_node_colors = {i: (0, 1, 0, 0) for i in range(len(node_attr))}
    blank_node_highlights = list(range(len(blank_node_colors)))

    # Create a new folder for this drug and cell line combination
    Path(plot_address+sample_info['drug_name'][0] + "_" + sample_info['cell_line_name'][0]).mkdir(parents=True, exist_ok=True)

    for feat in mol_feat_dict.keys():
        legend_pos_final = legend_pos + ", " + feat
        legend_neg_final = legend_neg + ", " + feat
        legends = [legend_pos_final, legend_neg_final]
        molsPerRow = 2
        subImgSize = (1000, 1000)
        nRows = 1
        drawOptions = None
        mols = [rdkit_mol, rdkit_mol]
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])

        if feat in ["Bond Type", "Conjugation", "Ring", "Stereo"]:
            # Dealing with edge features
            cur_neg_edge_mask = mask_molecules(neg_edge_mask, mol_feat_dict[feat][0], mol_feat_dict[feat][1])
            cur_pos_edge_mask = mask_molecules(pos_edge_mask, mol_feat_dict[feat][0], mol_feat_dict[feat][1])

            # for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
            neg_edge_mask_dict = aggregate_edge_directions(cur_neg_edge_mask, drug_graph)
            pos_edge_mask_dict = aggregate_edge_directions(cur_pos_edge_mask, drug_graph)

            neg_edge_colors = {i: (1, 0, 0, alpha) for i, alpha in enumerate(list(neg_edge_mask_dict.values()))}
            neg_edge_highlights = list(range(len(neg_edge_colors)))
            pos_edge_colors = {i: (0, 1, 0, alpha) for i, alpha in enumerate(list(pos_edge_mask_dict.values()))}
            pos_edge_highlights = list(range(len(pos_edge_colors)))

            highlightBondLists = [pos_edge_highlights, neg_edge_highlights]
            highlightBondColorsLists = [pos_edge_colors, neg_edge_colors]
            d2d.DrawMolecules(list(mols), legends=legends or None,
                              highlightBonds=highlightBondLists,
                              highlightBondColors=highlightBondColorsLists,
                              highlightAtoms=[blank_node_highlights, blank_node_highlights],
                              highlightAtomColors=[blank_node_colors, blank_node_colors])
            d2d.FinishDrawing()

        else:
            cur_neg_node_mask = mask_molecules(neg_node_mask, mol_feat_dict[feat][0], mol_feat_dict[feat][1])
            cur_pos_node_mask = mask_molecules(pos_node_mask, mol_feat_dict[feat][0], mol_feat_dict[feat][1])

            # Dealing with node features
            neg_node_mask_dict = aggregate_node(cur_neg_node_mask)
            pos_node_mask_dict = aggregate_node(cur_pos_node_mask)

            neg_atom_colors = {i: (1, 0, 0, alpha) for i, alpha in enumerate(list(neg_node_mask_dict.values()))}
            neg_atom_highlights = list(range(len(neg_atom_colors)))
            pos_atom_colors = {i: (0, 1, 0, alpha) for i, alpha in enumerate(list(pos_node_mask_dict.values()))}
            pos_atom_highlights = list(range(len(pos_atom_colors)))

            highlightAtomLists = [pos_atom_highlights, neg_atom_highlights]
            highlightAtomColorsLists = [pos_atom_colors, neg_atom_colors]

            d2d.DrawMolecules(list(mols), legends=legends or None,
                              highlightAtoms=highlightAtomLists,
                              highlightAtomColors=highlightAtomColorsLists,
                              highlightBonds=[blank_edge_highlights, blank_edge_highlights],
                              highlightBondColors=[blank_edge_colors, blank_edge_colors]
                              )
            d2d.FinishDrawing()

        address = plot_address + sample_info['drug_name'][
            0] + "_" + sample_info['cell_line_name'][0] + "/" + feat + "_full_plot.png"

        d2d.WriteDrawingText(address)

    # if drawOptions is not None:
    #   d2d.SetDrawOptions(drawOptions)
    # else:
    # dops = d2d.drawOptions()
    # for k, v in list(kwargs.items()):
    #   if hasattr(dops, k):
    #     setattr(dops, k, v)
    #     del kwargs[k]
    # d2d.DrawMolecules(list(mols), legends=legends or None,
    #                   highlightAtoms=highlightAtomLists,
    #                   highlightBonds=highlightBondLists,
    #                   highlightBondColors=highlightBondColorsLists,
    #                   highlightAtomColors=highlightAtomColorsLists)
    # d2d.FinishDrawing()
    # res = d2d.GetDrawingText()
    # address = '/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Plots/' + sample_info['drug_name'][
    #     0] + "_" + sample_info['cell_line_name'][0] + "_full_plot.png"
