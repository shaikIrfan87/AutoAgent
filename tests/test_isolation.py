import torch
import numpy as np
import pytest
from cognitive_engine.core.types import ExecutionContext
from cognitive_engine.core.consolidation import ConsolidationStore
from cognitive_engine.core.plastic_layer import FastPlasticLinear
from cognitive_engine.agent.orchestrator import CognitiveEngine
from self_eval_engine.credit import GraphCreditAssigner
from self_eval_engine.schemas import GraphTrajectory, CritiquePayload


class MockGraph:
    def __init__(self):
        self.edges = {}
        self.nodes = {"node_1": {"activation_threshold": 0.5, "base_threshold": 0.5}}

    def get_edge_weight(self, u: str, v: str) -> float:
        return self.edges.get((u, v), 0.0)

    def set_edge_weight(self, u: str, v: str, weight: float) -> None:
        self.edges[(u, v)] = weight


def test_consolidation_store_tenant_isolation(tmp_path):
    db_file = str(tmp_path / "tenant_memories.db")
    store = ConsolidationStore(db_path=db_file, dim=16)

    vec_a = np.ones(16, dtype=np.float32)
    vec_b = np.zeros(16, dtype=np.float32)
    vec_b[0] = 1.0

    # Tenant A writes confidential memory
    store.write_memory(
        content="Tenant A confidential strategy roadmap alpha",
        vector=vec_a,
        confidence=0.9,
        tenant_id="tenant_A",
        session_id="session_1",
    )

    # Tenant B writes separate memory
    store.write_memory(
        content="Tenant B public operations guide beta",
        vector=vec_b,
        confidence=0.9,
        tenant_id="tenant_B",
        session_id="session_2",
    )

    # Tenant B searches for Tenant A's confidential keyword
    results_b = store.hybrid_search(
        query_text="strategy roadmap alpha",
        query_vector=vec_a,
        top_k=5,
        tenant_id="tenant_B",
    )
    # Must return zero records or only Tenant B records
    assert len(results_b) == 0 or all(r[0].tenant_id == "tenant_B" for r in results_b)
    assert not any("Tenant A" in r[0].content for r in results_b)

    # Tenant A searches for own keyword
    results_a = store.hybrid_search(
        query_text="strategy roadmap alpha",
        query_vector=vec_a,
        top_k=5,
        tenant_id="tenant_A",
    )
    assert len(results_a) >= 1
    assert results_a[0][0].tenant_id == "tenant_A"
    assert "Tenant A" in results_a[0][0].content

    store.close()


def test_fast_plastic_linear_session_isolation():
    dim = 32
    layer = FastPlasticLinear(in_features=dim, out_features=dim)

    k = torch.randn(dim)
    v = torch.randn(dim)

    # Session 1 absorbs association
    layer.absorb(k, v, steps=3, tenant_id="tenant_1", session_id="session_A")

    out_s1 = layer.recall_fast(k, tenant_id="tenant_1", session_id="session_A")
    out_s2 = layer.recall_fast(k, tenant_id="tenant_1", session_id="session_B")
    out_other_tenant = layer.recall_fast(k, tenant_id="tenant_2", session_id="session_A")

    # Session 1 has strong recall
    assert torch.linalg.norm(out_s1).item() > 0.1

    # Session B and other tenant remain clean (uncontaminated)
    assert torch.linalg.norm(out_s2).item() == 0.0
    assert torch.linalg.norm(out_other_tenant).item() == 0.0

    # Penalizing Session 1 does not mutate Session B
    layer.penalize(k, v, penalty=0.2, tenant_id="tenant_1", session_id="session_A")
    assert torch.linalg.norm(layer.recall_fast(k, tenant_id="tenant_1", session_id="session_B")).item() == 0.0


def test_graph_credit_assigner_tenant_persistence(tmp_path):
    db_file = str(tmp_path / "tenant_graph.db")
    g_a = MockGraph()
    assigner_a = GraphCreditAssigner(g_a, db_path=db_file)

    # Tenant A sets edge
    g_a.set_edge_weight("node_1", "node_2", 0.75)
    assigner_a.save_state(tenant_id="tenant_A")

    # Tenant B sets same edge to different weight
    g_b = MockGraph()
    assigner_b = GraphCreditAssigner(g_b, db_path=db_file)
    g_b.set_edge_weight("node_1", "node_2", -0.40)
    assigner_b.save_state(tenant_id="tenant_B")

    # Reload Tenant A
    g_reload_a = MockGraph()
    assigner_reload_a = GraphCreditAssigner(g_reload_a, db_path=db_file)
    assigner_reload_a.load_state(tenant_id="tenant_A")
    assert assigner_reload_a.graph.get_edge_weight("node_1", "node_2") == pytest.approx(0.75)

    # Reload Tenant B
    g_reload_b = MockGraph()
    assigner_reload_b = GraphCreditAssigner(g_reload_b, db_path=db_file)
    assigner_reload_b.load_state(tenant_id="tenant_B")
    assert assigner_reload_b.graph.get_edge_weight("node_1", "node_2") == pytest.approx(-0.40)

    # Scoped dynamic edge queries
    assert assigner_reload_b.get_outgoing_edges("node_1", tenant_id="tenant_A") == {"node_2": pytest.approx(0.75)}
    assert assigner_reload_b.get_outgoing_edges("node_1", tenant_id="tenant_B") == {"node_2": pytest.approx(-0.40)}



def test_cognitive_engine_e2e_tenant_isolation(tmp_path):
    db_file = str(tmp_path / "engine_tenant.db")
    engine = CognitiveEngine(db_path=db_file)

    ctx_alpha = ExecutionContext(tenant_id="org_alpha", session_id="session_100")
    ctx_beta = ExecutionContext(tenant_id="org_beta", session_id="session_200")

    query = "Unique quantum calculation for organisation alpha"
    code = "print('Alpha output 999')"

    # Org Alpha runs deliberation
    res_alpha = engine.process(query, code_action=code, ctx=ctx_alpha)
    assert res_alpha["status"] == "system_2_success"

    # Org Beta executes same query - should NOT retrieve Alpha's memory in System 1
    # Because it is a novel query for Org Beta, or if it queries, Alpha's memory is filtered
    q_vec = engine.saliency._embed(query)
    search_beta = engine.consolidation.hybrid_search(
        query, q_vec, top_k=3, tenant_id=ctx_beta.tenant_id
    )
    assert len(search_beta) == 0

    # Org Alpha search finds it
    search_alpha = engine.consolidation.hybrid_search(
        query, q_vec, top_k=3, tenant_id=ctx_alpha.tenant_id
    )
    assert len(search_alpha) >= 1
    assert "Alpha output 999" in search_alpha[0][0].content

    engine.consolidation.close()


def test_plastic_weights_acid_persistence(tmp_path):
    db_file = str(tmp_path / "plastic_persist.db")
    ctx = ExecutionContext(tenant_id="tenant_omega", session_id="session_xyz")

    # 1. First run: engine absorbs association and snapshots working memory to SQLite
    engine1 = CognitiveEngine(db_path=db_file)
    k = torch.randn(engine1.dim)
    v = torch.randn(engine1.dim)
    engine1.plastic.absorb(k, v, steps=2, tenant_id=ctx.tenant_id, session_id=ctx.session_id)
    out1 = engine1.plastic.recall_fast(k, tenant_id=ctx.tenant_id, session_id=ctx.session_id)
    assert torch.linalg.norm(out1).item() > 0.1

    engine1.save_session_state(ctx)
    engine1.consolidation.close()

    # 2. Process restart: new engine instance reloads weights from SQLite
    engine2 = CognitiveEngine(db_path=db_file)
    loaded = engine2.load_session_state(ctx)
    assert loaded is True

    out2 = engine2.plastic.recall_fast(k, tenant_id=ctx.tenant_id, session_id=ctx.session_id)
    # Reconstructed recall matches original session recall
    assert torch.allclose(out1, out2, atol=1e-5)

    engine2.consolidation.close()


def test_prune_stale_plastic_weights(tmp_path):
    db_file = str(tmp_path / "prune_persist.db")
    engine = CognitiveEngine(db_path=db_file)
    ctx_stale = ExecutionContext(tenant_id="old_tenant", session_id="old_session")
    ctx_fresh = ExecutionContext(tenant_id="fresh_tenant", session_id="fresh_session")

    engine.save_session_state(ctx_stale)
    engine.save_session_state(ctx_fresh)

    # Manually backdate the old session to 14 days ago
    with engine.consolidation._lock, engine.consolidation.conn:
        engine.consolidation.conn.execute(
            "UPDATE plastic_weights SET last_updated = datetime('now', '-14 days') WHERE session_id = 'old_session'"
        )

    # Prune sessions older than 7 days
    pruned = engine.prune_stale_sessions(max_age_days=7)
    assert pruned == 1

    # Stale session is gone, fresh session remains
    assert engine.load_session_state(ctx_stale) is False
    assert engine.load_session_state(ctx_fresh) is True

    engine.consolidation.close()


def test_multi_turn_drift_and_hebbian_saturation(tmp_path):
    """50 alternating turns across 2 tenants: verifies zero cross-talk, retention, and bounded Hebbian weights."""
    db_file = str(tmp_path / "multi_turn_drift.db")
    engine = CognitiveEngine(db_path=db_file)
    ctx_a = ExecutionContext(tenant_id="tenant_alpha", session_id="session_drift_a")
    ctx_b = ExecutionContext(tenant_id="tenant_beta", session_id="session_drift_b")

    g_mock = MockGraph()
    assigner = GraphCreditAssigner(g_mock, db_path=db_file)

    torch.manual_seed(42)
    dim = engine.dim

    stored_a = []
    stored_b = []

    for turn in range(50):
        is_turn_a = (turn % 2 == 0)
        curr_ctx = ctx_a if is_turn_a else ctx_b
        t_id = curr_ctx.tenant_id

        # Unique orthogonal key per turn
        k = torch.zeros(dim)
        k[turn % dim] = 1.0
        v = torch.randn(dim)
        v = v / torch.linalg.norm(v)

        engine.plastic.absorb(k, v, steps=1, tenant_id=t_id, session_id=curr_ctx.session_id)
        if is_turn_a:
            stored_a.append((k, v))
        else:
            stored_b.append((k, v))

        # Continuous Hebbian updates alternating pass / fail
        traj = GraphTrajectory(id=f"traj_{turn}", reasoning_trace="", terminal_output="res", traversed_edges=[("node_X", "node_Y")])
        passed = (turn % 3 != 0)
        assigner.apply_feedback(traj, CritiquePayload(aggregate_score=0.9), passed=passed, tenant_id=t_id)

    # 1. Zero Cross-Talk: Tenant A keys queried under Tenant B session yield near-zero norm
    for k_a, _ in stored_a[:5]:
        out_b = engine.plastic.recall_fast(k_a, tenant_id=ctx_b.tenant_id, session_id=ctx_b.session_id)
        assert torch.linalg.norm(out_b).item() < 1e-4, "Cross-talk detected between tenants"

    # 2. Associative Retention: Tenant A recalls its own keys with positive cosine alignment
    for k_a, v_a in stored_a[-5:]:
        out_a = engine.plastic.recall_fast(k_a, tenant_id=ctx_a.tenant_id, session_id=ctx_a.session_id)
        cos_sim = torch.dot(out_a, v_a) / (torch.linalg.norm(out_a) * torch.linalg.norm(v_a) + 1e-9)
        assert cos_sim.item() > 0.40, "Catastrophic forgetting observed in fast weights"

    # 3. Dynamic Topology Bounds: Edge weights strictly bounded within [-1.0, 1.0]
    edges_a = assigner.get_outgoing_edges("node_X", tenant_id="tenant_alpha")
    edges_b = assigner.get_outgoing_edges("node_X", tenant_id="tenant_beta")
    assert -1.0 <= edges_a.get("node_Y", 0.0) <= 1.0
    assert -1.0 <= edges_b.get("node_Y", 0.0) <= 1.0

    # 4. Frobenius norm bounded under limit
    norm_a = torch.linalg.norm(engine.plastic._get_session_state(ctx_a.tenant_id, ctx_a.session_id), ord="fro").item()
    norm_b = torch.linalg.norm(engine.plastic._get_session_state(ctx_b.tenant_id, ctx_b.session_id), ord="fro").item()
    assert norm_a <= engine.plastic.frobenius_limit
    assert norm_b <= engine.plastic.frobenius_limit

    engine.consolidation.close()

