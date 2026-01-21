import numpy as np
import pandas as pd

np.random.seed(42)
n = 120

# Simulate true labels and predictions
references = np.random.choice([0, 1], size=n)
predictions = references.copy()

# Inject some errors
error_idx = np.random.choice(n, size=int(0.25 * n), replace=False)
predictions[error_idx] = 1 - predictions[error_idx]

# Simulate confidence
confidences = np.clip(
    np.random.normal(loc=0.75, scale=0.15, size=n),
    0.1, 0.99
)

# Reduce confidence on errors (realistic behavior)
confidences[error_idx] *= 0.8

# Simulate human trust (noisy tracking of confidence)
human_trust = np.clip(
    confidences + np.random.normal(0, 0.1, size=n),
    0.05, 0.99
)

# Simulate belief priors
belief_priors = np.clip(
    np.random.normal(0.4, 0.2, size=n),
    0.05, 0.95
)

# Simulate belief posteriors (shift toward model confidence)
belief_posteriors = np.clip(
    belief_priors + 0.5 * (confidences - belief_priors),
    0.05, 0.95
)

# Simulate explanation lengths (longer when uncertain)
explanation_length = np.clip(
    50 + (1 - confidences) * 120 + np.random.normal(0, 10, size=n),
    30, 200
)

df = pd.DataFrame({
    "prediction": predictions,
    "reference": references,
    "confidence": confidences.round(3),
    "human_trust": human_trust.round(3),
    "belief_prior": belief_priors.round(3),
    "belief_posterior": belief_posteriors.round(3),
    "explanation_length": explanation_length.astype(int),
})

df.to_csv("data/human_ai_trust_demo.csv", index=False)
print("Saved data/human_ai_trust_demo.csv")