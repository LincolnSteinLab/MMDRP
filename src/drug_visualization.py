from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from torch_geometric.utils import to_networkx


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
            edge_mask_dict[(u, v)] += float(val)
        # min-max scale to 0-1 range, want to use attribution as alpha while highlighting the molecule
        cur_min = min(edge_mask_dict.values())
        cur_max = max(edge_mask_dict.values())
        for key in edge_mask_dict.keys():
            edge_mask_dict[key] = (edge_mask_dict[key] - cur_min) / (cur_max - cur_min)

        return edge_mask_dict


def aggregate_node(node_mask):
    node_mask_dict = defaultdict(float)
    for row_i in list(range(node_mask.shape[0])):
        node_mask_dict[row_i] = float(node_mask[row_i,])

    cur_min = min(node_mask_dict.values())
    cur_max = max(node_mask_dict.values())
    for key in node_mask_dict.keys():
        node_mask_dict[key] = (node_mask_dict[key] - cur_min) / (cur_max - cur_min)

    return node_mask_dict


def drug_interpret_viz(edge_attr, node_attr, drug_graph, sample_info, plot_address):
    # edge_attr = zero_dl_attr_train[-1]
    # node_attr = zero_dl_attr_train[-2]

    neg_edge_mask = edge_attr.clamp(max=0)
    pos_edge_mask = edge_attr.clamp(min=0)
    neg_node_mask = node_attr.clamp(max=0)
    pos_node_mask = node_attr.clamp(min=0)

    # sum up across different node and edge feature attributions
    neg_edge_mask = neg_edge_mask.sum(axis=1)
    pos_edge_mask = pos_edge_mask.sum(axis=1)
    neg_node_mask = neg_node_mask.sum(axis=1)
    pos_node_mask = pos_node_mask.sum(axis=1)

    # for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
    neg_edge_mask_dict = aggregate_edge_directions(neg_edge_mask, drug_graph)
    pos_edge_mask_dict = aggregate_edge_directions(pos_edge_mask, drug_graph)
    neg_node_mask_dict = aggregate_node(neg_node_mask)
    pos_node_mask_dict = aggregate_node(pos_node_mask)

    rdkit_mol = Chem.MolFromSmiles(sample_info['smiles'][0])

    neg_edge_colors = {i: (1, 0, 0, alpha) for i, alpha in enumerate(list(neg_edge_mask_dict.values()))}
    neg_edge_highlights = list(range(len(neg_edge_colors)))
    pos_edge_colors = {i: (0, 1, 0, alpha) for i, alpha in enumerate(list(pos_edge_mask_dict.values()))}
    pos_edge_highlights = list(range(len(pos_edge_colors)))

    neg_atom_colors = {i: (1, 0, 0, alpha) for i, alpha in enumerate(list(neg_node_mask_dict.values()))}
    neg_atom_highlights = list(range(len(neg_atom_colors)))
    pos_atom_colors = {i: (0, 1, 0, alpha) for i, alpha in enumerate(list(pos_node_mask_dict.values()))}
    pos_atom_highlights = list(range(len(pos_atom_colors)))

    legend_pos = "Compound: " + sample_info['drug_name'][0] + ", Cell Line: " + sample_info['cell_line_name'][
        0] + " Positive Attributions"
    legend_neg = "Compound: " + sample_info['drug_name'][0] + ", Cell Line: " + sample_info['cell_line_name'][
        0] + " Negative Attributions"
    # d_neg = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # rdMolDraw2D.PrepareAndDrawMolecule(d_neg, rdkit_mol,
    #                                    highlightAtoms=neg_atom_highlights,
    #                                    highlightAtomColors=neg_atom_colors,
    #                                    highlightBonds=neg_edge_highlights,
    #                                    highlightBondColors=neg_edge_colors)
    # d_neg.FinishDrawing()
    # address = '/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Plots/' + sample_info['drug_name'][
    #     0] + "_" + sample_info['cell_line_name'][0] + "_neg_plot.png"
    # d_neg.WriteDrawingText(address)
    #
    # d_pos = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # rdMolDraw2D.PrepareAndDrawMolecule(d_pos, rdkit_mol,
    #                                    highlightAtoms=pos_atom_highlights,
    #                                    highlightAtomColors=pos_atom_colors,
    #                                    highlightBonds=pos_edge_highlights,
    #                                    highlightBondColors=pos_edge_colors)
    # d_pos.FinishDrawing()
    # address = '/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Plots/' + sample_info['drug_name'][
    #     0] + "_" + sample_info['cell_line_name'][0] + "_pos_plot.png"
    # d_pos.WriteDrawingText(address)

    # Chem.Draw.MolsToGridImage(
    #     [rdkit_mol, rdkit_mol],
    #     legends=[legend_pos, legend_neg],
    #     highlightAtomLists=[pos_atom_highlights, neg_atom_highlights],
    #     highlightBondLists=[pos_edge_highlights, neg_edge_highlights],
    #     highlightAtomColors=pos_edge_highlights,
    #     highlightBondColors=pos_edge_highlights,
    #     molsPerRow=2,
    #     subImgSize=(500, 500), useSVG=False)
    highlightAtomLists=[pos_atom_highlights, neg_atom_highlights]
    highlightBondLists=[pos_edge_highlights, neg_edge_highlights]
    highlightAtomColorsLists=[pos_atom_colors, neg_atom_colors]
    highlightBondColorsLists=[pos_edge_colors, neg_edge_colors]
    legends=[legend_pos, legend_neg]

    molsPerRow = 2
    subImgSize = (1000, 1000)
    nRows = 1
    drawOptions = None
    mols = [rdkit_mol, rdkit_mol]
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])

    # if drawOptions is not None:
    #   d2d.SetDrawOptions(drawOptions)
    # else:
    dops = d2d.drawOptions()
      # for k, v in list(kwargs.items()):
      #   if hasattr(dops, k):
      #     setattr(dops, k, v)
      #     del kwargs[k]
    d2d.DrawMolecules(list(mols), legends=legends or None,
                      highlightAtoms=highlightAtomLists,
                      highlightBonds=highlightBondLists,
                      highlightBondColors=highlightBondColorsLists,
                      highlightAtomColors=highlightAtomColorsLists)
    d2d.FinishDrawing()
    # res = d2d.GetDrawingText()
    # address = '/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Plots/' + sample_info['drug_name'][
    #     0] + "_" + sample_info['cell_line_name'][0] + "_full_plot.png"
    address = plot_address + sample_info['drug_name'][
        0] + "_" + sample_info['cell_line_name'][0] + "_full_plot.png"

    d2d.WriteDrawingText(address)
