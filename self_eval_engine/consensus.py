from collections import Counter
import hashlib
from typing import List, Tuple
from .schemas import Trajectory


class ConsensusEngine:
    @staticmethod
    def resolve_majority(trajectories: List[Trajectory]) -> Tuple[Trajectory, float]:
        """Groups candidates by terminal output to find consensus ratio."""
        if not trajectories:
            raise ValueError("No trajectories to evaluate.")

        answers = [t.terminal_output.strip() for t in trajectories]
        counts = Counter(answers)
        top_answers = counts.most_common()
        max_count = top_answers[0][1]
        tied = {ans for ans, count in top_answers if count == max_count}

        if len(tied) > 1:
            # Tie-break: highest verification score, lowest token entropy, deterministic sha256 invariant
            candidates = [t for t in trajectories if t.terminal_output.strip() in tied]
            candidates.sort(
                key=lambda t: (
                    -(t.verification.score if t.verification else 0.0),
                    t.token_entropy,
                    hashlib.sha256(t.terminal_output.strip().encode()).hexdigest(),
                )
            )
            winning_trajectory = candidates[0]
            winning_answer = winning_trajectory.terminal_output.strip()
        else:
            winning_answer = top_answers[0][0]
            winning_trajectory = next(
                t for t in trajectories if t.terminal_output.strip() == winning_answer
            )
        return winning_trajectory, max_count / len(trajectories)
