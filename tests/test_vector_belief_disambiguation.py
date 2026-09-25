import pytest
from cognitive_engine.core.saliency import VectorBeliefDisambiguator
from run import detect_ambiguity


def test_vector_belief_disambiguator_polysemy():
    dis = VectorBeliefDisambiguator()

    # Polysemous term 'mercury'
    res = dis.evaluate("Tell me about mercury")
    assert res["is_ambiguous"] is True
    assert res["topic"] == "mercury"
    assert len(res["options"]) >= 2

    # Unambiguous specific query
    res_unambig = dis.evaluate("Calculate the kinetic energy of a 1000kg vehicle")
    assert res_unambig["is_ambiguous"] is False


def test_dynamic_sense_registration():
    dis = VectorBeliefDisambiguator()
    # Register novel polysemous concept 'crane'
    dis.add_sense("crane", "Crane (bird)", "Large, long-legged and long-necked birds in the family Gruidae", "bird animal wildlife feathers wetlands nature biology")
    dis.add_sense("crane", "Crane (machine)", "Type of machine equipped with a hoist rope, wire ropes or chains and sheaves", "machine construction hoist engineering weight industrial lifting equipment")

    res = dis.evaluate("What is a crane?")
    assert res["is_ambiguous"] is True
    assert res["topic"] == "crane"
    assert len(res["options"]) == 2


def test_run_py_detect_ambiguity_integration():
    # Test that detect_ambiguity in run.py detects ambiguous terms via vector disambiguator
    res = detect_ambiguity("space")
    assert res["is_ambiguous"] is True
    assert "options" in res
    assert len(res["options"]) >= 2
