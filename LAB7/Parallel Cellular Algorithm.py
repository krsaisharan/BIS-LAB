import numpy as np

def sphere(x):
    return np.sum(np.square(x))

def get_neighbors(grid, i, j):
    rows, cols = grid.shape
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = (i + di) % rows, (j + dj) % cols
            neighbors.append(grid[ni, nj])
    return neighbors

def parallel_cellular_algorithm(func, dim=2, grid_size=(10, 10), iterations=100, learning_rate=0.3, mutation_rate=0.1):
    rows, cols = grid_size
    grid = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            position = np.random.uniform(-5.12, 5.12, dim)
            fitness = func(position)
            grid[i, j] = {'position': position, 'fitness': fitness}

    global_best = min((grid[i, j] for i in range(rows) for j in range(cols)), key=lambda c: c['fitness'])

    for it in range(iterations):
        new_grid = np.copy(grid)
        for i in range(rows):
            for j in range(cols):
                cell = grid[i, j]
                neighbors = get_neighbors(grid, i, j)
                best_neighbor = min(neighbors, key=lambda c: c['fitness'])
                new_pos = cell['position'] + learning_rate * (best_neighbor['position'] - cell['position']) + mutation_rate * np.random.uniform(-1, 1, dim)
                new_fit = func(new_pos)
                if new_fit < cell['fitness']:
                    new_grid[i, j] = {'position': new_pos, 'fitness': new_fit}
        grid = new_grid
        current_best = min((grid[i, j] for i in range(rows) for j in range(cols)), key=lambda c: c['fitness'])
        if current_best['fitness'] < global_best['fitness']:
            global_best = current_best
        if it % 10 == 0 or it == iterations - 1:
            print(f"Iteration {it+1}/{iterations} - Best Fitness: {global_best['fitness']:.6f}")

    print("\nBest solution found:")
    print("Position:", global_best['position'])
    print("Fitness:", global_best['fitness'])
    return global_best

best = parallel_cellular_algorithm(sphere, dim=2, grid_size=(10, 10), iterations=100)
