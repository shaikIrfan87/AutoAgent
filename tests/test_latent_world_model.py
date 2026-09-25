import torch
from cognitive_engine.core.world_model import LatentWorldModel
from cognitive_engine.agent.program_synthesizer import MCTSProgramSynthesizer


def test_latent_world_model_forward_and_residual_transition():
    wm = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=32)
    s_0 = torch.randn(14)

    # Test encoder
    z = wm.encode(s_0)
    assert z.shape == (32,)

    # Test single-step forward residual transition
    z_next, reward, risk = wm(z, a_idx=3)
    assert z_next.shape == (32,)
    assert reward.dim() == 0 or reward.numel() == 1
    assert 0.0 <= float(risk.item()) <= 1.0


def test_latent_world_model_training_step():
    wm = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=32)
    s_t = torch.randn(14)
    s_tp1 = torch.randn(14)

    loss = wm.train_transition_step(
        s_t=s_t,
        a_idx=5,
        s_tp1=s_tp1,
        reward_target=0.85,
        risk_target=0.10,
        lr=0.01,
    )
    assert loss > 0.0
    assert not torch.isnan(torch.tensor(loss))


def test_latent_world_model_multi_step_imagination():
    wm = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=32)
    s_0 = torch.randn(14)
    actions = [0, 2, 4, 1]

    tot_reward, max_risk = wm.imagine_rollout(s_0, actions)
    assert isinstance(tot_reward, float)
    assert isinstance(max_risk, float)
    assert 0.0 <= max_risk <= 1.0


def test_mcts_synthesizer_latent_world_model_integration():
    synthesizer = MCTSProgramSynthesizer(max_depth=3, max_expansions=80, use_neural_guidance=True)
    assert hasattr(synthesizer, "world_model")
    assert isinstance(synthesizer.world_model, LatentWorldModel)

    # Test training transition step on synthesizer
    s_t = torch.randn(14)
    s_tp1 = torch.randn(14)
    loss = synthesizer.train_world_model_step(s_t, 1, s_tp1, reward_target=1.0, risk_target=0.05)
    assert loss > 0.0

    # Test program synthesis with latent imagination active
    task = [(((1, 2), (3, 4)), ((2, 1), (4, 3)))]  # flip_h
    prog = synthesizer.synthesize(task)
    assert prog is not None
    assert prog(((1, 2), (3, 4))) == ((2, 1), (4, 3))
