import numpy as np
from k_center import two_approx_k_center_interval_refinement, two_approx_k_center_max_distance, calculate_radius
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time

def run_experiments(X, k, num_runs=30):
    results_matrix = np.zeros((num_runs, 4))  # Matriz para armazenar raio, silhueta, ARI e tempo de execução

    for i in range(num_runs):
        start_time = time.time()

        # Executando o algoritmo 2-aproximado
        centers_interval = two_approx_k_center_interval_refinement(X, k)

        # Calculando o raio da solução
        radius = calculate_radius(X, centers_interval)
        
        # Calculando as métricas de avaliação
        labels = assign_labels(X, centers_interval)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(_, labels)  # _ deve ser substituído por rótulos verdadeiros se disponíveis

        runtime = time.time() - start_time

        # Armazenando os resultados na matriz
        results_matrix[i, 0] = radius
        results_matrix[i, 1] = silhouette
        results_matrix[i, 2] = ari
        results_matrix[i, 3] = runtime

    return results_matrix

def assign_labels(X, centers):
    """
    Atribui rótulos aos pontos com base no centro mais próximo.
    """
    labels = np.zeros(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        distances = [np.linalg.norm(x - center) for center in centers]
        labels[i] = np.argmin(distances)
    return labels

# Criação dos dados para testes
X, _ = make_blobs(n_samples=700, centers=5, n_features=2, random_state=42)
k = 5

# Executando os experimentos
results_matrix = run_experiments(X, k)

# Exibindo os resultados
print(results_matrix)
print('------------------------------------\n\n')
print('std Radius: ', np.std(results_matrix[:,0]))
print('mean Radius: ', np.mean(results_matrix[:,0]))
print('\n\n')
print('std silhouette: ', np.std(results_matrix[:,1]))
print('mean silhouette: ', np.mean(results_matrix[:,1]))
print('\n\n')
print('std ARI: ', np.std(results_matrix[:,2]))
print('mean ARI: ', np.mean(results_matrix[:,2]))
print('\n\n')
print('std runtime: ', np.std(results_matrix[:,3]))
print('mean runtime: ', np.mean(results_matrix[:,3]))
print('\n\n')
print('------------------------------------\n\n')
