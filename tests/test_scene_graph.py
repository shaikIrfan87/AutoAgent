import pytest
from cognitive_engine.core.dsl import to_grid
from cognitive_engine.core.scene_graph import SceneGraphExtractor, GridObject


def test_extract_objects():
    # Two disjoint objects: blue 2x2 square and red 1x1 dot
    g = to_grid([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 2],
    ])
    objects = SceneGraphExtractor.extract_objects(g)
    assert len(objects) == 2

    # Verify attributes
    square = next(o for o in objects if o.color == 1)
    dot = next(o for o in objects if o.color == 2)
    assert square.size == 4
    assert square.bbox == (0, 1, 0, 1)
    assert dot.size == 1
    assert dot.bbox == (2, 2, 3, 3)


def test_build_scene_graph_relations():
    # Three objects: A above B, B left of C
    g = to_grid([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 4],
    ])
    graph = SceneGraphExtractor.build_scene_graph(g)
    assert graph.number_of_nodes() == 3

    # Check node attributes
    nodes_by_color = {graph.nodes[n]["color"]: n for n in graph.nodes}
    assert 4 in nodes_by_color
    assert 3 in nodes_by_color

    # Verify spatial edges
    edges = list(graph.edges(data=True))
    relations = [d["relation"] for _, _, d in edges]
    assert "above" in relations
    assert "below" in relations
    assert "left_of" in relations
    assert "right_of" in relations
    assert "same_color" in relations
