import random, math

hospital = (50, 50)

def generate_patients(n, area=100):
    return [hospital] + [(random.randint(0, area), random.randint(0, area)) for _ in range(n)]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def ambulance_route(locations, ants=10, iters=50, alpha=1, beta=5, rho=0.5, q=100):
    n = len(locations)
    dists = [[dist(locations[i], locations[j]) for j in range(n)] for i in range(n)]
    pher = [[1 for _ in range(n)] for _ in range(n)]
    best_tour, best_len = None, float('inf')

    for _ in range(iters):
        tours = []
        for _ in range(ants):
            tour = [0]
            while len(tour) < n:
                curr = tour[-1]
                choices = [j for j in range(1, n) if j not in tour]
                if not choices:
                    break
                probs = [(pher[curr][j] ** alpha) * ((1 / dists[curr][j]) ** beta) for j in choices]
                total = sum(probs)
                probs = [p / total for p in probs]
                tour.append(random.choices(choices, probs)[0])
            tour.append(0)
            length = sum(dists[tour[i]][tour[i+1]] for i in range(len(tour)-1))
            if length < best_len:
                best_tour, best_len = tour, length
            tours.append((tour, length))

        for i in range(n):
            for j in range(n):
                pher[i][j] *= (1 - rho)
        for tour, length in tours:
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i+1]
                pher[a][b] += q / length
                pher[b][a] += q / length

    return best_tour, best_len

if __name__ == "__main__":
    patients = generate_patients(n=8)
    tour, length = ambulance_route(patients, ants=20, iters=100)
    print(round(length, 2))
    print([f'H' if i == 0 else f'P{i}' for i in tour])
