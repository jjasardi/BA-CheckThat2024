import numpy as np
from itertools import combinations

def mean_pairwise_disagreement(disagreement_matrix, model_indices):
    """
    Calculate the mean pairwise disagreement for a given set of model indices.
    
    :param disagreement_matrix: A 2D numpy array representing the disagreement fractions between models.
    :param model_indices: A list of indices representing the models in the group.
    :return: The mean pairwise disagreement for the given set of models.
    """
    pairwise_disagreements = []
    for (i, j) in combinations(model_indices, 2):
        pairwise_disagreements.append(disagreement_matrix[i, j])
    mean_pairwise_disagreement = np.mean(pairwise_disagreements)    
    return mean_pairwise_disagreement

def get_all_combinations_with_disagreement(disagreement_matrix):
    """
    Generate a list of all possible combinations of models with their mean pairwise disagreement.
    
    :param disagreement_matrix: A 2D numpy array representing the disagreement fractions between models.
    :return: A list of tuples, each containing a combination of models and their mean pairwise disagreement, sorted by the disagreement.
    """
    num_models = disagreement_matrix.shape[0]
    all_combinations = []

    for r in range(2, num_models + 1):
        for model_combination in combinations(range(num_models), r):
            mean_disagreement = mean_pairwise_disagreement(disagreement_matrix, model_combination)
            all_combinations.append((model_combination, mean_disagreement))

    # Sort the combinations by mean pairwise disagreement in descending order
    all_combinations.sort(key=lambda x: x[1], reverse=True)
    
    return all_combinations

# Example usage:
# Assume we have a 4x4 disagreement fraction matrix
disagreement_matrix = np.array([
    [0.0, 0.2, 0.3, 0.4],
    [0.2, 0.0, 0.5, 0.1],
    [0.3, 0.5, 0.0, 0.6],
    [0.4, 0.1, 0.6, 0.0]
])

all_combinations = get_all_combinations_with_disagreement(disagreement_matrix)

# Print the sorted combinations with their mean pairwise disagreements
for combination, mean_disagreement in all_combinations:
    print(f"Models: {combination}, Mean Pairwise Disagreement: {mean_disagreement:.4f}")
