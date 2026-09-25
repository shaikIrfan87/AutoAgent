import os
import tempfile
import torch
from cognitive_engine.agent.executive_loop import ExecutiveLoop
from cognitive_engine.agent.program_synthesizer import (
    MCTSProgramSynthesizer,
    NeuralPolicyValueNetwork,
)
from cognitive_engine.core.plastic_layer import TTTAttentionLayer
from cognitive_engine.core.scene_graph import (
    ContinuousRelationalEncoder,
    GridObject,
    SceneGraphExtractor,
)


def test_neural_policy_value_mcts_guidance_and_training():
    net = NeuralPolicyValueNetwork(num_ops=22, feature_dim=32)
    grid_sample = [(((1, 2), (3, 4)),)]
    feats = net.extract_features(grid_sample[0])
    assert feats.shape == (1, 14)

    logits, value = net(feats)
    assert logits.shape == (1, 22)
    assert -1.0 <= float(value.item()) <= 1.0

    # Test real gradient optimization step
    loss = net.train_step(
        states=[grid_sample[0], grid_sample[0]],
        target_action_indices=[0, 1],
        target_values=[0.8, 0.4],
        lr=0.01,
    )
    assert loss > 0.0

    # Test MCTS Program Synthesizer with neural guidance
    mcts = MCTSProgramSynthesizer(max_depth=3, max_expansions=100, use_neural_guidance=True)
    task = [(((1, 2), (3, 4)), ((2, 1), (4, 3)))]  # flip_h
    prog = mcts.synthesize(task)
    assert prog is not None
    assert prog(((1, 2), (3, 4))) == ((2, 1), (4, 3))


def test_continuous_relational_gnn_encoder():
    encoder = ContinuousRelationalEncoder(hidden_dim=32)
    grid = ((1, 1, 0, 0), (0, 0, 2, 2))
    objects = SceneGraphExtractor.extract_objects(grid)
    assert len(objects) == 2

    affinities = encoder(objects)
    assert "spatial" in affinities
    assert "contact" in affinities
    assert "color" in affinities
    assert affinities["spatial"].shape == (2, 2)

    # Build continuous scene graph
    cg = SceneGraphExtractor.build_continuous_scene_graph(grid, encoder=encoder, threshold=0.1)
    assert len(cg.nodes) == 2
    assert len(cg.edges) > 0
    for _, _, d in cg.edges(data=True):
        assert "affinity" in d
        assert 0.0 <= d["affinity"] <= 1.0


def test_ttt_attention_layer_online_adaptation():
    layer = TTTAttentionLayer(embed_dim=32, num_heads=2, eta=0.1, frobenius_limit=2.5)
    x = torch.randn(2, 6, 32)

    out, norm = layer(x, adapt_online=True)
    assert out.shape == (2, 6, 32)
    assert norm > 0.0
    assert norm <= 2.5

    # Test reset
    layer.reset_ttt()
    assert float(torch.linalg.norm(layer.W_ttt).item()) == 0.0


def test_executive_loop_open_world_grounding():
    loop = ExecutiveLoop()

    # 1. System inspection
    sys_info = loop.execute_open_world_action("system_inspect", {})
    assert sys_info["success"] is True
    assert "os" in sys_info
    assert sys_info["cpu_count"] > 0

    # 2. OS shell command
    shell_res = loop.execute_open_world_action("shell", {"command": "echo AGI_GROUNDING_TEST"})
    assert shell_res["success"] is True
    assert "AGI_GROUNDING_TEST" in shell_res["stdout"]

    # 3. Filesystem write and read
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, "test_ground.txt")
        w_res = loop.execute_open_world_action(
            "filesystem_write",
            {"path": test_file, "content": "grounded_autonomous_state"},
        )
        assert w_res["success"] is True
        assert w_res["bytes_written"] == len("grounded_autonomous_state")

        r_res = loop.execute_open_world_action("filesystem_read", {"path": test_file})
        assert r_res["success"] is True
        assert r_res["content"] == "grounded_autonomous_state"


def test_curiosity_online_neural_learning_and_mcp_open_world():
    from cognitive_engine.agent.orchestrator import CognitiveEngine
    from cognitive_engine.core.mcp_server import MCPServer

    engine = CognitiveEngine()
    assert hasattr(engine, "ttt_attention")

    # Verify curiosity step triggers autotelic experiment with input_grid and neural training
    mock_memory = [
        {"id": "concept_test_1", "confidence": 0.40, "last_accessed": 0.0, "content": "test rule"}
    ]
    engine.autotelic.scan_epistemic_gaps(mock_memory, {})
    res = engine.curiosity.step()
    assert res is not None
    assert getattr(res, "success", False) is True
    assert getattr(res, "input_grid", None) is not None

    # Verify MCP server open-world action tool
    mcp = MCPServer(engine)
    assert "execute_open_world_action" in mcp.tools

    mcp_res = mcp._handle_open_world_action({
        "action_type": "system_inspect",
        "params": {},
    })
    assert mcp_res["isError"] is False
    assert "content" in mcp_res


