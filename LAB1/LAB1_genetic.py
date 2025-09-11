import random
import math

def fitness_function(x):
    return x * math.sin(10 * math.pi * x) + 1.0

POPULATION_SIZE = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 50
CHROMOSOME_LENGTH = 16

def decode(chromosome):
    decimal_value = int(chromosome, 2)
    max_value = 2**CHROMOSOME_LENGTH - 1
    return decimal_value / max_value

def create_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))
        population.append(chromosome)
    return population

def evaluate_population(population):
    fitnesses = []
    for chromosome in population:
        x = decode(chromosome)
        fitnesses.append(fitness_function(x))
    return fitnesses

def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_indices = random.choices(range(len(population)), weights=selection_probs, k=2)
    return population[selected_indices[0]], population[selected_indices[1]]

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(chromosome):
    new_chromosome = ''
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            new_chromosome += '1' if bit == '0' else '0'
        else:
            new_chromosome += bit
    return new_chromosome

def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        fitnesses = evaluate_population(population)
        max_fitness_gen = max(fitnesses)
        max_index = fitnesses.index(max_fitness_gen)
        if max_fitness_gen > best_fitness:
            best_fitness = max_fitness_gen
            best_solution = population[max_index]
        print(f"Generation {gen+1} - Best Fitness: {best_fitness:.5f}")
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])
        population = new_population[:POPULATION_SIZE]

    best_x = decode(best_solution)
    return best_x, best_fitness

best_x, best_val = genetic_algorithm()
print(f"\nBest solution found: x = {best_x:.5f}, f(x) = {best_val:.5f}")
