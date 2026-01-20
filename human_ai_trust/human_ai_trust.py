"""Human Trust & Uncertainty Metrics for AI Evaluation.

This local metric exists so `evaluate.load("human_ai_trust")` can work when
running tests from a source checkout (it loads metrics from `./<name>/<name>.py`).
"""

from typing import Any, Dict, List, Optional

import datasets
import evaluate
import numpy as np


_DESCRIPTION = """
Human Trust & Uncertainty Metrics for AI Evaluation.

This metric suite operationalizes trust calibration, belief updating,
and uncertainty alignment for human–AI interaction evaluation.
It complements traditional performance metrics by surfacing
human-centered signals about trust, belief dynamics, and confidence communication.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (List[Any]):
        Model predictions.
    references (List[Any]):
        Ground truth labels.
    confidences (List[float]):
        Model confidence values in [0, 1].
    human_trust_scores (List[float]):
        Human trust ratings in [0, 1].
    belief_priors (Optional[List[float]]):
        User beliefs before seeing AI output.
    belief_posteriors (Optional[List[float]]):
        User beliefs after seeing AI output.
    explanation_complexity (Optional[List[float]]):
        Explanation complexity scores (e.g., length, entropy, readability).

Returns:
    Dict[str, float]:
        A dictionary containing:
            - expected_trust_error
            - trust_sensitivity_index
            - belief_shift_magnitude (optional)
            - overconfidence_penalty
            - overconfidence_penalty_normalized
            - explanation_confidence_alignment (optional)
"""


def _safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class HumanAITrust(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                    "confidences": datasets.Value("float32"),
                    "human_trust_scores": datasets.Value("float32"),
                }
            ),
            reference_urls=[],
        )

    def _validate_inputs(
        self,
        predictions: List[Any],
        references: List[Any],
        confidences: List[float],
        human_trust_scores: List[float],
        belief_priors: Optional[List[float]],
        belief_posteriors: Optional[List[float]],
        explanation_complexity: Optional[List[float]],
    ) -> None:
        n = len(predictions)
        if not (len(references) == n and len(confidences) == n and len(human_trust_scores) == n):
            raise ValueError("All required input lists must have equal length.")

        if belief_priors is not None and len(belief_priors) != n:
            raise ValueError("belief_priors must have the same length as predictions.")

        if belief_posteriors is not None and len(belief_posteriors) != n:
            raise ValueError("belief_posteriors must have the same length as predictions.")

        if explanation_complexity is not None and len(explanation_complexity) != n:
            raise ValueError("explanation_complexity must have the same length as predictions.")

        for c in confidences:
            if not (0.0 <= c <= 1.0):
                raise ValueError("All confidence values must be in [0, 1].")

        for t in human_trust_scores:
            if not (0.0 <= t <= 1.0):
                raise ValueError("All human trust scores must be in [0, 1].")

        if belief_priors is not None:
            for b in belief_priors:
                if not (0.0 <= b <= 1.0):
                    raise ValueError("All belief_priors values must be in [0, 1].")

        if belief_posteriors is not None:
            for b in belief_posteriors:
                if not (0.0 <= b <= 1.0):
                    raise ValueError("All belief_posteriors values must be in [0, 1].")

    def _compute(
        self,
        predictions: List[Any],
        references: List[Any],
        confidences: List[float],
        human_trust_scores: List[float],
        belief_priors: Optional[List[float]] = None,
        belief_posteriors: Optional[List[float]] = None,
        explanation_complexity: Optional[List[float]] = None,
    ) -> Dict[str, Optional[float]]:
        self._validate_inputs(
            predictions,
            references,
            confidences,
            human_trust_scores,
            belief_priors,
            belief_posteriors,
            explanation_complexity,
        )

        conf_arr = np.asarray(confidences, dtype=float)
        trust_arr = np.asarray(human_trust_scores, dtype=float)

        # ETE
        ete = _safe_mean(np.abs(trust_arr - conf_arr))

        # TSI
        tsi = _safe_corr(trust_arr, conf_arr)

        # OCP
        errors = np.asarray([pred != ref for pred, ref in zip(predictions, references)], dtype=float)
        ocp = _safe_mean(conf_arr * errors)

        # OCP normalized
        mean_conf = _safe_mean(conf_arr)
        ocp_norm = float(ocp / mean_conf) if mean_conf > 0.0 else 0.0

        belief_shift_magnitude: Optional[float] = None
        if belief_priors is not None and belief_posteriors is not None:
            pri = np.asarray(belief_priors, dtype=float)
            post = np.asarray(belief_posteriors, dtype=float)
            belief_shift_magnitude = _safe_mean(np.abs(post - pri))

        explanation_confidence_alignment: Optional[float] = None
        if explanation_complexity is not None:
            comp = np.asarray(explanation_complexity, dtype=float)
            explanation_confidence_alignment = _safe_corr(comp, conf_arr)

        return {
            "expected_trust_error": float(ete),
            "trust_sensitivity_index": float(tsi),
            "belief_shift_magnitude": belief_shift_magnitude,
            "overconfidence_penalty": float(ocp),
            "overconfidence_penalty_normalized": float(ocp_norm),
            "explanation_confidence_alignment": explanation_confidence_alignment,
        }
