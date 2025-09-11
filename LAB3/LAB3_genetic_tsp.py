import random
import numpy as np

# Generate distance matrix for n cities
def generate_distance_matrix(n, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n, 2) * 100
    return np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)), coords

# Calculate total path distance
def path_distance(path, dist_matrix):
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + dist_matrix[path[-1], path[0]]

# Order Crossover (OX)
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    hole = [item for item in parent2 if item not in parent1[a:b]]
    return hole[:a] + parent1[a:b] + hole[a:]

# Swap Mutation
def mutate(path):
    a, b = random.sample(range(len(path)), 2)
    path[a], path[b] = path[b], path[a]

# Genetic Algorithm for TSP
def gene_expression_tsp(n_cities=10, pop_size=50, generations=200, mutation_rate=0.1):
    dist_matrix, coords = generate_distance_matrix(n_cities)

    # Initialize population
    population = [random.sample(range(n_cities), n_cities) for _ in range(pop_size)]
    fitness = [1 / path_distance(p, dist_matrix) for p in population]
    best_solution = population[np.argmax(fitness)]
    best_fitness = max(fitness)

    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            # Tournament Selection
            parents = random.sample(population, 5)
            parents.sort(key=lambda p: path_distance(p, dist_matrix))
            parent1, parent2 = parents[0], parents[1]

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                mutate(child)

            new_population.append(child)

        # Evaluate new fitness
        population = new_population
        fitness = [1 / path_distance(p, dist_matrix) for p in population]
        current_best_idx = np.argmax(fitness)

        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx]

    return best_solution, 1 / best_fitness, coords

# Run GEA
solution, distance, coords = gene_expression_tsp()
print("GEA Best Path:", solution)
print("GEA Best Distance:", distance)
