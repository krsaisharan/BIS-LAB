import numpy as np


def sphere(x):
    return np.sum(x**2)

best_pos, best_score = gwo(sphere, dim=5, lb=-10, ub=10, num_wolves=10, max_iter=50)
print("Best position:", best_pos)
print("Best score:", best_score)

def gwo(objective_function, dim, lb, ub, num_wolves, max_iter):
    positions = np.random.uniform(lb, ub, (num_wolves, dim))
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_score = float('inf')
    delta_score = float('inf')
    
    for t in range(max_iter):
        for i in range(num_wolves):
            fitness = objective_function(positions[i])
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i].copy()
                
        a = 2 - t * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                X2 = beta_pos[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i][j] = (X1 + X2 + X3) / 3
                
            positions[i] = np.clip(positions[i], lb, ub)
            
    return alpha_pos, alpha_score
