#assisted with AI 

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

def load_data(filepath="../data/plant_knowledge.csv"):
    """
    Loads the CSV data and returns a NumPy array of responses (X),
    number of informants (N), and number of questions (M), excludes 'Informant' column.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values # excludes 'Informant' column
    N, M = X.shape
    return X, N, M

def build_cultural_consensus_model(X, N, M):
    """
    Constructs the Bayesian model for cultural consensus
    """
    with pm.Model() as model:
        # Prior for D_i ∈ [0.5, 1] — scale Beta(1, 1) to [0.5, 1]
        raw_D = pm.Beta("raw_D", alpha=1, beta=1, shape=N)
        D = pm.Deterministic("D", 0.5 + 0.5 * raw_D)  # scales to [0.5, 1]

        # Prior for consensus answers Z_j ∈ {0, 1}
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Broadcast D and Z to compute p_ij
        D_reshaped = D[:, None]  # shape (N, 1)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)  # shape (N, M)

        # Likelihood: observed responses
        pm.Bernoulli("X_obs", p=p, observed=X)

    return model

if __name__ == "__main__":
    X, N, M = load_data() # load data
    model = build_cultural_consensus_model(X, N, M) # build model

    # Perform inference 
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True
        )

    # Analyze results 
    summary = az.summary(trace, var_names=["D", "Z"])
    print("\nPosterior Summary:")
    print(summary)

    # Competence Estimates
    D_means = trace.posterior["D"].mean(dim=("chain", "draw")).values.flatten()
    print("\nPosterior Mean Competence per Informant:")
    for i, d in enumerate(D_means, start=1):
        print(f"P{i}: {d:.3f}")

    most = np.argmax(D_means)
    least = np.argmin(D_means)
    print(f"\nMost competent: P{most+1} ({D_means[most]:.3f})")
    print(f"Least competent: P{least+1} ({D_means[least]:.3f})")

    az.plot_posterior(trace, var_names=["D"])

    # Consensus Answer Estimates
    Z_means = trace.posterior["Z"].mean(dim=("chain", "draw")).values.flatten()
    Z_map = (Z_means > 0.5).astype(int)
    print("\nConsensus Answer Key (CCT):")
    print(Z_map)

    az.plot_posterior(trace, var_names=["Z"])

    # Compare with Majority Vote
    majority_vote = (X.mean(axis=0) > 0.5).astype(int)
    print("\nNaive Majority Vote Answer Key:")
    print(majority_vote)

    print("\nComparison (CCT vs Majority Vote):")
    for j, (cct, naive) in enumerate(zip(Z_map, majority_vote), start=1):
        match = "✔️" if cct == naive else "❌"
        print(f"Q{j}: CCT = {cct}, Naive = {naive} {match}")
