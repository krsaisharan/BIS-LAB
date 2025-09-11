import numpy as np

def sphere(x):
    return sum(xi ** 2 for xi in x)

num_particles = 30
num_dimensions = 2
max_iter = 100
w = 0.5            
c1 = 1.5           
c2 = 1.5           
bounds = [-10, 10] 

positions = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
velocities = np.zeros((num_particles, num_dimensions))
pbest = positions.copy()
pbest_fitness = np.array([sphere(p) for p in positions])

gbest_idx = np.argmin(pbest_fitness)
gbest = pbest[gbest_idx].copy()
gbest_fitness = pbest_fitness[gbest_idx]

for iteration in range(max_iter):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()

        velocities[i] = (
            w * velocities[i] +
            c1 * r1 * (pbest[i] - positions[i]) +
            c2 * r2 * (gbest - positions[i])
        )

        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        fitness = sphere(positions[i])
        if fitness < pbest_fitness[i]:
            pbest[i] = positions[i]
            pbest_fitness[i] = fitness

    gbest_idx = np.argmin(pbest_fitness)
    if pbest_fitness[gbest_idx] < gbest_fitness:
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

print("PSO Best Solution:", gbest)
print("PSO Best Fitness:", gbest_fitness)
