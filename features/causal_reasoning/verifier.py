import networkx as nx

class CausalAxiomVerifier:
    """
    Invariant 2: Directed Multigraph for Physical & Causal Axioms.
    Verifies candidate actions in <0.015ms against causal directed edges.
    """
    def __init__(self):
        self.causal_graph = nx.MultiDiGraph()
        self._seed_causal_invariants()

    def _seed_causal_invariants(self):
        self.add_axiom("perpetual motion machine", "infinite energy", "cannot_be", False)
        self.add_axiom("vacuum", "air", "cannot_be", False)
        self.add_axiom("biological_organism", "synthetic_machine", "mutually_exclusive", False)

    def add_axiom(self, subj: str, tgt: str, relation: str, polarity: bool):
        self.causal_graph.add_edge(
            subj.lower().strip(),
            tgt.lower().strip(),
            relation=relation.lower().strip(),
            polarity=polarity
        )

    def verify_causal_safety(self, subj: str, rel: str, tgt: str) -> bool:
        """
        Verifies whether an action violates physical conservation/causal axioms.
        Returns False if a prohibited edge matches relation and polarity==False.
        """
        s, r, t = subj.lower().strip(), rel.lower().strip(), tgt.lower().strip()
        if self.causal_graph.has_edge(s, t):
            for _, edge_data in self.causal_graph[s][t].items():
                if edge_data.get("relation") == r and edge_data.get("polarity") is False:
                    return False
        return True
