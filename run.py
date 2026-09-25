#!/usr/bin/env python3
"""Unified AutoAgent Entry Point.

Run directly:
    python run.py              # Interactive CLI session
    python run.py "your query" # Single query execution
    python run.py --autonomous # 24/7 background self-learning daemon
    python run.py --test       # Run live verification tests
    python run.py --server     # Start FastAPI / MCP background daemon
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure root workspace is on python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ambiguity resolution is dynamically grounded in episodic memory (consolidation.py)


DISAMBIGUATION_MAP: Dict[str, Dict[str, str]] = {
    "space": {
        "1": "Outer space",
        "2": "Vector space",
    },
    "time travel": {
        "1": "Time travel",
        "2": "Time travel in fiction",
    },
    "mercury": {
        "1": "Mercury (planet)",
        "2": "Mercury (element)",
        "3": "Freddie Mercury",
    },
    "matrix": {
        "1": "Matrix (mathematics)",
        "2": "The Matrix",
    },
    "apple": {
        "1": "Apple Inc.",
        "2": "Apple",
    },
}


def check_conversational_or_identity(query: str) -> Optional[str]:
    """Returns instant response for identity, greeting, dialogue, and introspection queries."""
    from cognitive_engine.agent.orchestrator import handle_direct_dialogue, check_introspection
    return handle_direct_dialogue(query) or check_introspection(query)


_vector_disambiguator = None


def get_vector_disambiguator():
    global _vector_disambiguator
    if _vector_disambiguator is None:
        try:
            from cognitive_engine.core.saliency import VectorBeliefDisambiguator
            _vector_disambiguator = VectorBeliefDisambiguator()
        except Exception:
            _vector_disambiguator = None
    return _vector_disambiguator


RUN_COMPOUND_EXCLUSIONS = {
    "space": ["disk space", "storage space", "memory space", "free space", "drive space", "drive c"],
    "python": ["python file", "python files", "python script", "python scripts", "python -c", "python code", ".py", "in python"],
}


def detect_ambiguity(query: str, consolidation_store: Optional[Any] = None) -> Dict[str, Any]:
    """Checks if query is ambiguous via vector belief space or episodic memory traces."""
    clean_q = query.strip().lower().rstrip("?").strip()
    clean_q = re.sub(r"\s+", " ", clean_q)

    action_prefixes = ("check ", "find ", "list ", "run ", "show ", "get ", "calculate ", "verify ")
    if any(clean_q.startswith(p) for p in action_prefixes):
        return {"is_ambiguous": False}

    for word, exclusions in RUN_COMPOUND_EXCLUSIONS.items():
        if word in clean_q and any(ex in clean_q for ex in exclusions):
            return {"is_ambiguous": False}

    # 1. Metric vector belief disambiguation
    dis = get_vector_disambiguator()
    if dis:
        try:
            res = dis.evaluate(query)
            if res.get("is_ambiguous"):
                return res
        except Exception:
            pass

    # 2. Episodic memory store disambiguation traces
    if consolidation_store is not None:
        try:
            traces = consolidation_store.hybrid_search(f"disambiguation: {clean_q}", top_k=2)
            if traces:
                options = [t["content"] for t in traces if "content" in t]
                if options:
                    return {"is_ambiguous": True, "topic": clean_q, "options": options}
        except Exception:
            pass

    return {"is_ambiguous": False}






def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI + MCP server."""
    import uvicorn
    print(f"Starting AutoAgent Daemon at http://{host}:{port} ...")
    uvicorn.run("cognitive_engine.api:app", host=host, port=port, reload=False)


def run_tests():
    """Run the live cognitive test suite."""
    from scripts.run_live_test import run_cognitive_live_test
    print("Executing AutoAgent Live Verification Suite...\n")
    run_cognitive_live_test()


def handle_manual_command(cmd: str) -> Optional[bool]:
    """Handles reserved user commands. Returns False to exit, True if handled, None if unknown."""
    c = cmd.strip().lower()
    EXIT_VARIANTS = {"exit", "quit", "q", "ezit", "exut", "exi", "ext", "quti", "qiut", "exitt", "quitt"}
    if c in EXIT_VARIANTS:
        return False
    if c == "help":
        print("\nCommands:")
        print("  <query>  : Send query (auto-researches web & executes in sandbox)")
        print("  test     : Run live end-to-end cognitive tests")
        print("  demo     : Run self-evaluation engine demo")
        print("  server   : Launch the HTTP / WebSocket / MCP API server")
        print("  clear    : Clear terminal screen")
        print("  exit     : Shut down and exit\n")
        return True
    if c == "test":
        from scripts.run_live_test import run_cognitive_live_test
        run_cognitive_live_test()
        return True
    if c == "demo":
        from scripts.run_self_eval_demo import main as run_demo
        run_demo()
        return True
    if c == "server":
        print("Launching server (press Ctrl+C to return to CLI)...")
        try:
            start_server()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        return True
    if c == "clear":
        os.system("cls" if os.name == "nt" else "clear")
        return True
    return None


import threading

# Ensure UTF-8 output encoding and ANSI escape support on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if os.name == "nt":
    os.system("")


class StatusTracker:
    """Ephemeral animated status indicator without multi-line console noise."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stage = "thinking"
        # Braille spinner if UTF-8, else universal ASCII slash spinner
        if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
            self._spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        else:
            self._spinner = ["|", "/", "-", "\\"]

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r\033[2K\033[36m{self._spinner[idx]} [{self._stage}...]\033[0m ")
            sys.stdout.flush()
            idx = (idx + 1) % len(self._spinner)
            time.sleep(0.08)
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def set_stage(self, stage: str):
        self._stage = stage

    def start(self, initial_stage: str = "thinking"):
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return
        self._stage = initial_stage
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=0.5)
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def interactive_cli():
    """Run an interactive command-line session."""
    from cognitive_engine.agent.orchestrator import CognitiveEngine

    print("=" * 65)
    print("                AutoAgent Cognitive Runtime")
    print("=" * 65)
    print("Initializing subsystems (Saliency, Causal Graph, Sandbox)...")

    t0 = time.time()
    db_file = str(ROOT_DIR / "assets" / "cognitive_memory.db")
    engine = CognitiveEngine(db_path=db_file)
    if hasattr(engine, "curiosity") and engine.curiosity:
        engine.curiosity.interval_sec = 20.0
        engine.curiosity.start()
    try:
        from cognitive_engine.core.continuous_learner import ContinuousCoreEvolver
        evolver = ContinuousCoreEvolver(engine, check_interval_sec=30.0)
        evolver.start()
    except Exception:
        evolver = None
    tracker = StatusTracker()
    init_time = (time.time() - t0) * 1000
    print(f"System ready in {init_time:.1f}ms.")
    print("Commands: 'test', 'demo', 'help', 'exit'. All other tasks execute autonomously.\n")

    session_state: Dict[str, Any] = {}

    try:
        while True:
            try:
                user_input = input("AutoAgent> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            # Check for reserved user commands first
            EXIT_VARIANTS = {"exit", "quit", "q", "ezit", "exut", "exi", "ext", "quti", "qiut", "exitt", "quitt"}
            c_input = user_input.strip().lower()
            if c_input in EXIT_VARIANTS or c_input in ["help", "test", "demo", "server", "clear"]:
                status = handle_manual_command(user_input)
                if status is False:
                    print("Exiting...")
                    break
                continue

            # 1. Clarification unwrap
            if session_state.get("awaiting_clarification"):
                clarification = session_state.pop("awaiting_clarification")
                choice = user_input.strip()
                topic = clarification.get("topic", "")

                target_query = None
                if topic in DISAMBIGUATION_MAP and choice in DISAMBIGUATION_MAP[topic]:
                    target_query = DISAMBIGUATION_MAP[topic][choice]
                elif choice.isdigit():
                    idx = int(choice) - 1
                    opts = clarification.get("options", [])
                    if 0 <= idx < len(opts):
                        target_query = opts[idx]
                else:
                    target_query = choice

                if target_query:
                    clean_concept = re.sub(
                        r"^(Are you referring to|Or are you asking about|Or do you mean)\s*",
                        "",
                        target_query,
                        flags=re.IGNORECASE,
                    )
                    clean_concept = re.split(r"/|\bor do you mean\b", clean_concept, flags=re.IGNORECASE)[0].strip()
                    clean_concept = clean_concept.split("(")[0].split("?")[0].strip()
                    target_query = clean_concept or target_query
                    orig_q = clarification.get("original_query", "")
                    if orig_q and len(orig_q.split()) > 2 and topic in orig_q.lower():
                        target_query = f"{orig_q} ({clean_concept})"
                    elif len(target_query.split()) <= 1:
                        target_query = f"Explain {clean_concept}"
                else:
                    target_query = choice

                if hasattr(engine, "curiosity") and engine.curiosity:
                    engine.curiosity.pause()
                tracker.start("thinking")
                try:
                    result = engine.process_interactive(target_query, status_cb=tracker.set_stage)
                finally:
                    tracker.stop()
                    if hasattr(engine, "curiosity") and engine.curiosity:
                        engine.curiosity.resume()
                print(f"{result}\n")
                continue

            # 2. Check for Ambiguity
            ambiguity = detect_ambiguity(user_input)
            if ambiguity["is_ambiguous"]:
                session_state["awaiting_clarification"] = {
                    "original_query": user_input,
                    "topic": ambiguity["topic"],
                    "options": ambiguity["options"],
                }
                print(f"\nYour query '{user_input}' can refer to multiple topics:")
                for i, opt in enumerate(ambiguity["options"], 1):
                    print(f"  [{i}] {opt}")
                print("Which one are you referring to?\n")
                continue

            # 3. Direct Dialogue
            dialogue_ans = check_conversational_or_identity(user_input)
            if dialogue_ans:
                print(f"{dialogue_ans}\n")
                continue

            # 4. Streamlined Execution with Status Indicator
            if hasattr(engine, "curiosity") and engine.curiosity:
                engine.curiosity.pause()
            tracker.start("thinking")
            try:
                result = engine.process_interactive(user_input, status_cb=tracker.set_stage)
            finally:
                tracker.stop()
                if hasattr(engine, "curiosity") and engine.curiosity:
                    engine.curiosity.resume()

            print(f"{result}\n")

    except KeyboardInterrupt:
        print("\nSession interrupted.")
    finally:
        try:
            if evolver:
                evolver.stop()
            if hasattr(engine, "curiosity") and engine.curiosity:
                engine.curiosity.stop()
            engine.sandbox.close()
            engine.consolidation.close()
        except Exception:
            pass
        print("AutoAgent shutdown complete.")


def main():
    args = sys.argv[1:]

    if not args:
        interactive_cli()
        return

    first_arg = args[0].strip()

    if first_arg in ("--help", "-h"):
        print(__doc__)
    elif first_arg in ("--autonomous", "--life", "-a"):
        from cognitive_engine.agent.orchestrator import CognitiveEngine
        db_file = str(ROOT_DIR / "assets" / "cognitive_memory.db")
        try:
            engine = CognitiveEngine(db_path=db_file)
        except Exception as e:
            print(f"[FATAL] CognitiveEngine failed to initialize: {e}", file=sys.stderr)
            sys.exit(1)
        if not engine.curiosity:
            print("[FATAL] CuriosityDaemon did not initialize. Check sub-component imports.", file=sys.stderr)
            sys.exit(1)
        print("AutoAgent Autonomous Life Daemon active.")
        print("Continuous cycle: Epistemic Scanner -> Hypothesis -> Sandbox (Delta S) -> Memory Consolidation -> RSI.")
        print("Press Ctrl+C to stop.\n")
        engine.curiosity.interval_sec = 5.0
        engine.curiosity.start()
        if hasattr(engine, "core_evolver") and engine.core_evolver:
            engine.core_evolver.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Autonomous Life Daemon...")
        finally:
            if hasattr(engine, "core_evolver") and engine.core_evolver:
                engine.core_evolver.stop()
            engine.curiosity.stop()
            engine.sandbox.close()
            engine.consolidation.close()
            print("Shutdown complete.")
    elif first_arg in ("--server", "-s"):
        start_server()
    elif first_arg in ("--test", "-t"):
        run_tests()
    elif first_arg in ("--demo", "-d"):
        from scripts.run_self_eval_demo import main as run_demo
        run_demo()
    else:
        # Execute single query directly from CLI args
        query = " ".join(args)

        # Check identity responses first
        identity_ans = check_conversational_or_identity(query)
        if identity_ans:
            print(identity_ans)
            return

        # Check ambiguity in single-shot mode
        ambiguity = detect_ambiguity(query)
        if ambiguity["is_ambiguous"]:
            print(f"Your query '{query}' can refer to multiple topics:")
            for i, opt in enumerate(ambiguity["options"], 1):
                print(f"  [{i}] {opt}")
            print("Please specify context (e.g. run interactive mode 'python run.py').")
            return

        from cognitive_engine.agent.orchestrator import CognitiveEngine
        db_file = str(ROOT_DIR / "assets" / "cognitive_memory.db")
        engine = CognitiveEngine(db_path=db_file)
        tracker = StatusTracker()
        tracker.start("thinking")
        try:
            result = engine.process_interactive(query, status_cb=tracker.set_stage)
        finally:
            tracker.stop()
        print(result)
        engine.sandbox.close()
        engine.consolidation.close()


if __name__ == "__main__":
    main()
