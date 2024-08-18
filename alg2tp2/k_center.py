import numpy as np

def calculate_radius(X, centers):
    """
    Calcula o raio da solução, que é a distância máxima do ponto mais distante ao seu centro mais próximo.
    """
    max_distance = 0
    for x in X:
        min_distance = min(np.linalg.norm(x - center) for center in centers)
        max_distance = max(max_distance, min_distance)
    return max_distance

def two_approx_k_center_interval_refinement(X, k, epsilon=0.05):
    """
    Implementa o algoritmo 2-aproximado com refinamento de intervalo para o problema de k-centros.
    """
    lower_bound, upper_bound = 0, np.max([np.linalg.norm(x - y) for x in X for y in X])
    
    while upper_bound - lower_bound > epsilon:
        mid = (lower_bound + upper_bound) / 2
        centers = farthest_first_traversal(X, k)
        radius = calculate_radius(X, centers)
        
        if radius <= mid:
            upper_bound = mid
        else:
            lower_bound = mid
    
    return centers

def two_approx_k_center_max_distance(X, k):
    """
    Implementa o algoritmo 2-aproximado que maximiza a distância entre os centros.
    """
    centers = [X[np.random.randint(len(X))]]
    
    while len(centers) < k:
        distances = np.array([min([np.linalg.norm(x - center) for center in centers]) for x in X])
        new_center = X[np.argmax(distances)]
        centers.append(new_center)
    
    return centers

def farthest_first_traversal(X, k):
    """
    Seleciona os centros iniciais usando a estratégia de Farthest First Traversal.
    """
    centers = [X[np.random.randint(len(X))]]
    for _ in range(1, k):
        distances = np.min([np.linalg.norm(X - center, axis=1) for center in centers], axis=0)
        new_center = X[np.argmax(distances)]
        centers.append(new_center)
    return centers
