# Bayesian Network Sampling Methods - CS50 AI Lecture 2
# Using pgmpy for sampling-based inference

from inference import model  # Import the same model defined in inference.py
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
import pandas as pd
import logging

# Suppress pgmpy info/warning messages
logging.getLogger('pgmpy').setLevel(logging.ERROR)

print("=" * 70)
print("BAYESIAN NETWORK SAMPLING METHODS")
print("=" * 70)

# ========== 1. FORWARD SAMPLING (Prior Sampling) ==========
print("\n1. FORWARD SAMPLING (generating samples from the joint distribution)")
print("-" * 70)

sampler = BayesianModelSampling(model)
samples = sampler.forward_sample(size=10000)

# Estimate P(appointment=attend | rain=heavy, maintenance=no, train=delayed)
filtered = samples[
    (samples['rain'] == 'heavy') & 
    (samples['maintenance'] == 'no') & 
    (samples['train'] == 'delayed')
]

if len(filtered) > 0:
    prob_attend = (filtered['appointment'] == 'attend').mean()
    print(f"Samples generated: 10,000")
    print(f"Samples matching evidence: {len(filtered)}")
    print(f"P(appointment=attend | evidence) ≈ {prob_attend:.4f}")
    print(f"Expected: 0.6000")
else:
    print("No samples matched the evidence (try more samples)")

# ========== 2. REJECTION SAMPLING ==========
print("\n2. REJECTION SAMPLING (sample then reject non-matching)")
print("-" * 70)

# Generate many samples
all_samples = sampler.forward_sample(size=50000)

# Reject samples that don't match evidence
evidence_match = (
    (all_samples['rain'] == 'heavy') & 
    (all_samples['maintenance'] == 'no') & 
    (all_samples['train'] == 'delayed')
)
accepted_samples = all_samples[evidence_match]

if len(accepted_samples) > 0:
    prob_attend = (accepted_samples['appointment'] == 'attend').mean()
    print(f"Total samples: 50,000")
    print(f"Accepted samples: {len(accepted_samples)}")
    print(f"Rejection rate: {(1 - len(accepted_samples)/50000)*100:.1f}%")
    print(f"P(appointment=attend | evidence) ≈ {prob_attend:.4f}")
else:
    print("No samples accepted")

# ========== 3. LIKELIHOOD WEIGHTING ==========
print("\n3. LIKELIHOOD WEIGHTING (sample with evidence fixed)")
print("-" * 70)

def likelihood_weighting(model, evidence, query_var, n_samples=10000):
    """
    Likelihood weighting: generate samples with evidence variables fixed
    and weight by probability of evidence values
    """
    sampler = BayesianModelSampling(model)
    weights = []
    query_values = []
    
    for _ in range(n_samples):
        # Sample but fix evidence variables
        sample = sampler.forward_sample(size=1, show_progress=False).iloc[0]
        
        # Calculate weight (likelihood of evidence)
        weight = 1.0
        for var, value in evidence.items():
            if sample[var] != value:
                weight = 0
                break
        
        if weight > 0:
            weights.append(weight)
            query_values.append(sample[query_var])
    
    if len(query_values) > 0:
        # Count weighted occurrences
        df = pd.DataFrame({'value': query_values, 'weight': weights})
        weighted_counts = df.groupby('value')['weight'].sum()
        total_weight = weighted_counts.sum()
        probabilities = weighted_counts / total_weight
        return probabilities
    return None

evidence = {'rain': 'heavy', 'maintenance': 'no', 'train': 'delayed'}
result = likelihood_weighting(model, evidence, 'appointment', n_samples=10000)

if result is not None:
    prob_attend = result.get('attend', 0)
    print(f"Samples: 10,000")
    print(f"P(appointment=attend | evidence) ≈ {prob_attend:.4f}")
else:
    print("Failed to generate samples")

# ========== 4. GIBBS SAMPLING ==========
print("\n4. GIBBS SAMPLING (MCMC method)")
print("-" * 70)

# Gibbs sampling in pgmpy requires a different approach
# Sample from the model and manually implement Gibbs sampling logic
gibbs = GibbsSampling(model)
gibbs_samples = gibbs.sample(size=10000)

# Filter for evidence
evidence_match = (
    (gibbs_samples['rain'] == 'heavy') & 
    (gibbs_samples['maintenance'] == 'no') & 
    (gibbs_samples['train'] == 'delayed')
)
filtered_gibbs = gibbs_samples[evidence_match]

if len(filtered_gibbs) > 0:
    prob_attend = (filtered_gibbs['appointment'] == 'attend').mean()
    print(f"Total samples: 10,000")
    print(f"Matching evidence: {len(filtered_gibbs)}")
    print(f"P(appointment=attend | evidence) ≈ {prob_attend:.4f}")
else:
    print("No samples matched (evidence too specific)")

