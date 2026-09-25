"""
Object-Centric Entity-Relation Scene Graph Extractor for Visual Reasoning.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import torch
import torch.nn as nn

try:
    from .dsl import Grid, find_connected_components, to_grid
except ImportError:
    from core.dsl import Grid, find_connected_components, to_grid


@dataclass
class GridObject:
    object_id: int
    color: int
    size: int
    bbox: Tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    centroid: Tuple[float, float]
    pixels: Set[Tuple[int, int]]


class SceneGraphExtractor:
    """Parses 2D integer grids into attributed multi-object entity-relation graphs."""

    @staticmethod
    def extract_objects(g: Grid) -> List[GridObject]:
        """Extract connected components into typed spatial objects."""
        components = find_connected_components(g, monochromatic=True)
        objects = []
        for idx, comp in enumerate(components):
            rows = [r for r, c in comp]
            cols = [c for r, c in comp]
            sample_r, sample_c = next(iter(comp))
            color = g[sample_r][sample_c]
            bbox = (min(rows), max(rows), min(cols), max(cols))
            centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
            objects.append(GridObject(
                object_id=idx,
                color=color,
                size=len(comp),
                bbox=bbox,
                centroid=centroid,
                pixels=comp,
            ))
        return objects

    @classmethod
    def build_scene_graph(cls, g: Grid) -> nx.MultiDiGraph:
        """Constructs NetworkX MultiDiGraph with spatial and topological entity relations."""
        objects = cls.extract_objects(g)
        graph = nx.MultiDiGraph()

        for obj in objects:
            graph.add_node(
                obj.object_id,
                color=obj.color,
                size=obj.size,
                bbox=obj.bbox,
                centroid=obj.centroid,
            )

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                a_id, b_id = obj_a.object_id, obj_b.object_id
                cr_a, cc_a = obj_a.centroid
                cr_b, cc_b = obj_b.centroid

                # Spatial directions
                if cr_a < cr_b:
                    graph.add_edge(a_id, b_id, relation="above")
                elif cr_a > cr_b:
                    graph.add_edge(a_id, b_id, relation="below")
                if cc_a < cc_b:
                    graph.add_edge(a_id, b_id, relation="left_of")
                elif cc_a > cc_b:
                    graph.add_edge(a_id, b_id, relation="right_of")

                # Color & size
                if obj_a.color == obj_b.color:
                    graph.add_edge(a_id, b_id, relation="same_color")
                if obj_a.size > obj_b.size:
                    graph.add_edge(a_id, b_id, relation="larger_than")
                elif obj_a.size < obj_b.size:
                    graph.add_edge(a_id, b_id, relation="smaller_than")

                # Topological contact
                b_pixels = set(obj_b.pixels)
                touches = any(
                    (r + dr, c + dc) in b_pixels
                    for r, c in obj_a.pixels
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                if touches:
                    graph.add_edge(a_id, b_id, relation="touches")

        return graph

    @staticmethod
    def render_objects(objects: List[GridObject], shape: Tuple[int, int], bg: int = 0) -> Grid:
        """Render a list of GridObjects back into an immutable Grid."""
        h, w = shape
        mat = [[bg] * w for _ in range(h)]
        for obj in objects:
            for r, c in obj.pixels:
                if 0 <= r < h and 0 <= c < w:
                    mat[r][c] = obj.color
        return tuple(tuple(row) for row in mat)

    @classmethod
    def remove_isolated_objects(cls, g: Grid) -> Grid:
        """Remove objects that have no contact adjacency with any other object."""
        objects = cls.extract_objects(g)
        if len(objects) <= 1:
            return g
        graph = cls.build_scene_graph(g)
        connected_ids = {u for u, v, d in graph.edges(data=True) if d.get("relation") == "touches"}
        connected_ids |= {v for u, v, d in graph.edges(data=True) if d.get("relation") == "touches"}
        kept = [obj for obj in objects if obj.object_id in connected_ids]
        return cls.render_objects(kept, (len(g), len(g[0])))

    @classmethod
    def recolor_touching_objects(cls, g: Grid, touch_color: int, new_color: int) -> Grid:
        """Change color of objects touching touch_color to new_color."""
        objects = cls.extract_objects(g)
        graph = cls.build_scene_graph(g)
        target_ids = {u for u in graph.nodes if graph.nodes[u]["color"] == touch_color}
        touching_ids = set()
        for u in target_ids:
            for _, v, d in graph.out_edges(u, data=True):
                if d.get("relation") == "touches":
                    touching_ids.add(v)
            for v, _, d in graph.in_edges(u, data=True):
                if d.get("relation") == "touches":
                    touching_ids.add(v)
        h, w = len(g), len(g[0])
        mat = [list(row) for row in g]
        for obj in objects:
            if obj.object_id in touching_ids:
                for r, c in obj.pixels:
                    mat[r][c] = new_color
        return tuple(tuple(row) for row in mat)

    @classmethod
    def build_continuous_scene_graph(
        cls,
        g: Grid,
        encoder: Optional["ContinuousRelationalEncoder"] = None,
        threshold: float = 0.5,
    ) -> nx.MultiDiGraph:
        """Constructs NetworkX MultiDiGraph with continuous relational affinities using a neural encoder."""
        if encoder is None:
            encoder = ContinuousRelationalEncoder()

        objects = cls.extract_objects(g)
        graph = nx.MultiDiGraph()
        if not objects:
            return graph

        for obj in objects:
            graph.add_node(
                obj.object_id,
                color=obj.color,
                size=obj.size,
                bbox=obj.bbox,
                centroid=obj.centroid,
            )

        with torch.no_grad():
            affinities = encoder(objects)

        n = len(objects)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a_id, b_id = objects[i].object_id, objects[j].object_id

                # Latent relational edge predictions from neural heads
                if "spatial" in affinities:
                    aff_val = float(affinities["spatial"][i, j].item())
                    if aff_val >= threshold:
                        graph.add_edge(a_id, b_id, relation="continuous_spatial", affinity=round(aff_val, 4))

                if "contact" in affinities:
                    aff_val = float(affinities["contact"][i, j].item())
                    if aff_val >= threshold:
                        graph.add_edge(a_id, b_id, relation="continuous_contact", affinity=round(aff_val, 4))

                if "color" in affinities:
                    aff_val = float(affinities["color"][i, j].item())
                    if aff_val >= threshold:
                        graph.add_edge(a_id, b_id, relation="continuous_color_match", affinity=round(aff_val, 4))

        return graph


class ContinuousRelationalEncoder(nn.Module):
    """Continuous Relational GNN Encoder for discovering latent entity-relation topologies.
    Maps spatial and color attributes of objects to continuous edge affinities.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Node features: normalized centroid (2), bbox (4), size (1), color one-hot (10) = 17
        self.node_embed = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_spatial = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_contact = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_color = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    @staticmethod
    def object_to_features(obj: GridObject, max_dim: float = 30.0) -> torch.Tensor:
        """Extract continuous feature vector for an attributed GridObject."""
        cr, cc = obj.centroid
        min_r, max_r, min_c, max_c = obj.bbox
        spatial_feats = [
            cr / max_dim,
            cc / max_dim,
            min_r / max_dim,
            max_r / max_dim,
            min_c / max_dim,
            max_c / max_dim,
            float(obj.size) / (max_dim * max_dim),
        ]
        color_one_hot = [1.0 if i == obj.color else 0.0 for i in range(10)]
        return torch.tensor(spatial_feats + color_one_hot, dtype=torch.float32)

    def forward(self, objects: List[GridObject]) -> Dict[str, torch.Tensor]:
        """Compute continuous affinity matrices between all object pairs."""
        if not objects:
            return {}
        feats = torch.stack([self.object_to_features(obj) for obj in objects])
        h = self.node_embed(feats)  # (N, hidden_dim)

        spatial_aff = torch.sigmoid(torch.matmul(torch.matmul(h, self.W_spatial), h.t()))
        contact_aff = torch.sigmoid(torch.matmul(torch.matmul(h, self.W_contact), h.t()))
        color_aff = torch.sigmoid(torch.matmul(torch.matmul(h, self.W_color), h.t()))
        return {
            "spatial": spatial_aff,
            "contact": contact_aff,
            "color": color_aff,
        }


