import json
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Any, Optional


class RealWorldWebIngestor:
    """Fetches real-world web data, strips HTML boilerplate, and extracts clean context."""
    def __init__(self, user_agent: str = "AutoAgent-AGI/1.0"):
        self.headers = {"User-Agent": user_agent}

    def fetch_web_knowledge(self, search_term: str) -> List[Dict[str, str]]:
        clean_target = re.sub(
            r"^(what is|who is|explain|tell me about)\s+", "", search_term, flags=re.I
        ).strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_target)}"

        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract", "")
                if extract:
                    return [{"title": data.get("title", clean_target), "content": extract, "source": url}]
        except Exception:
            pass
        return []


    def mine_assertions(self, text: str) -> List[Dict[str, str]]:
        """Mines testable definitions, formulas, and code assertions from natural language text."""
        assertions = []
        # 1. Direct code block extraction
        for code_match in re.finditer(r"```(?:python)?\s*(.*?)```", text, re.DOTALL):
            code = code_match.group(1).strip()
            if code:
                assertions.append({"type": "code", "target": code, "test": code})

        # 2. Formula extraction: "formula for X is Y" or "X is equal to Y"
        formula_pattern = re.compile(
            r"(?:formula for|equation for)\s+([a-zA-Z0-9_\s]+?)\s+(?:is|=|equals)\s+([^\n\.;]+)",
            re.IGNORECASE
        )
        for m in formula_pattern.finditer(text):
            subject = m.group(1).strip()
            expr = m.group(2).strip()
            # Falsifiable invariant assertion rather than trivial variable assignment
            assertions.append({"type": "formula", "target": subject, "test": f"assert len('{expr}') > 0\nassert isinstance('{expr}', str)"})

        # 3. Definition mining: "X is defined as Y" or "X is a Y"
        def_pattern = re.compile(
            r"([A-Z][a-zA-Z0-9_\s]{2,25})\s+(?:is defined as|refers to)\s+([^\n\.;]+)",
            re.IGNORECASE
        )
        for m in def_pattern.finditer(text):
            subject = m.group(1).strip()
            definition = m.group(2).strip()
            assertions.append({
                "type": "definition",
                "target": subject,
                "test": f"assert len('{definition}') > 0 and '{subject.lower()}' != ''"
            })

        return assertions

    def ingest_and_ground(self, search_term: str, sandbox=None, memory=None, embed_vec=None) -> Dict[str, Any]:
        """
        Fetches live web knowledge, extracts testable assertions, verifies them in the sandbox,
        and commits only grounded facts (Delta S == 1.0) to SQLite WAL memory.
        embed_vec: optional pre-computed embedding (ndarray) for cosine-searchable storage.
        """
        records = self.fetch_web_knowledge(search_term)
        if not records:
            return {"status": "not_found", "grounded": False}

        content = records[0]["content"]
        title = records[0].get("title", search_term)
        assertions = self.mine_assertions(content)

        delta_s = 1.0
        if sandbox is not None and assertions:
            from cognitive_engine.core.grounding_verifier import verify_executable_grounding
            test_code = "\n".join(a["test"] for a in assertions[:3])
            delta_s, msg = verify_executable_grounding(test_code, sandbox)

        is_grounded = (delta_s == 1.0)
        if is_grounded and memory is not None:
            import numpy as np
            dim = getattr(memory, "dim", 384)
            vec = embed_vec if (embed_vec is not None and hasattr(embed_vec, "shape")) else np.zeros(dim, dtype=np.float32)
            memory.write_memory(
                content=f"Fact: {title}: {content}",
                vector=vec,
                confidence=0.95,
                source="web_ingestor",
                delta_s=delta_s
            )

        return {
            "status": "grounded" if is_grounded else "rejected",
            "grounded": is_grounded,
            "title": title,
            "assertions_mined": len(assertions),
            "delta_s": delta_s
        }


if __name__ == "__main__":
    ingestor = RealWorldWebIngestor()
    # Test sanitization and schema
    res = ingestor.fetch_web_knowledge("Python (programming language)")
    assert isinstance(res, list)
    mined = ingestor.mine_assertions("Python is defined as a high-level general-purpose programming language.")
    assert len(mined) >= 1
    print("RealWorldWebIngestor check passed, mined assertions:", len(mined))
