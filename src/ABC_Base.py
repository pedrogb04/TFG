import numpy as np


class ABC_Base:
    def __init__(self, objective_function, r_dim, d_dim, boundaries, limit=5):
        self.objective_function = objective_function
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.real_boundaries = boundaries["real"]
        self.discrete_boundaries = boundaries["discrete"]
        self.limit = limit

        self.solution_history = None
        self.convergence = None

    def get_initial_population(self, population_size):
        lower_boundary_r, upper_boundary_r = self.real_boundaries
        pop_r = np.random.uniform(
            low=lower_boundary_r,
            high=upper_boundary_r,
            size=(population_size, self.r_dim),
        )

        lower_boundary_d, upper_boundary_d = self.discrete_boundaries
        pop_d = np.random.randint(
            low=lower_boundary_d,
            high=upper_boundary_d + 1,
            size=(population_size, self.d_dim),
        )

        for j in range(population_size):
            while sum(pop_d[j]) == 0:
                pop_d[j] = np.random.randint(
                    low=lower_boundary_d, high=upper_boundary_d + 1, size=self.d_dim
                )

        accuracy = np.zeros(population_size, dtype=float)
        num_features = np.zeros(population_size, dtype=float)
        trials = np.zeros(population_size, dtype=int)
        fitness = np.zeros(population_size, dtype=float)

        initial_pop = {
            "real": pop_r,
            "discrete": pop_d,
            "accuracy": accuracy,
            "num_features": num_features,
            "trials": trials,
            "fitness": fitness,
        }
        for i in range(population_size):
            sol = self.select_solution(initial_pop, i)
            (
                initial_pop["fitness"][i],
                initial_pop["num_features"][i],
                initial_pop["accuracy"][i],
            ) = self.objective_function(sol)

        return initial_pop

    def perturb_solution(self, population, population_size, bee_to_perturb):
        k = bee_to_perturb
        while k == bee_to_perturb:
            k = np.random.randint(0, population_size)

        new_solution = self.select_solution(population, bee_to_perturb)

        # Mutation of real variables (Hyperparameters)
        phi_r = np.random.uniform(low=-1, high=1, size=self.r_dim)
        new_solution["real"] = population["real"][bee_to_perturb] + phi_r * (
            population["real"][bee_to_perturb] - population["real"][k]
        )
        # Apply limits
        new_solution["real"] = np.clip(
            new_solution["real"],
            self.real_boundaries[0],
            self.real_boundaries[1],
        )

        # Mutation of discrete variables (Features)
        phi_d = np.random.rand(self.d_dim)
        move_probs = phi_d * (
            np.abs(population["discrete"][bee_to_perturb] - population["discrete"][k])
        )

        rand = np.random.rand(self.d_dim)

        flip_mask = rand < move_probs

        new_solution["discrete"][flip_mask] ^= 1

        if sum(new_solution["discrete"]) == 0:
            new_solution["discrete"][np.random.randint(0, self.d_dim)] = 1

        (
            new_solution["fitness"],
            new_solution["num_features"],
            new_solution["accuracy"],
        ) = self.objective_function(new_solution)
        new_solution["trials"] = 0

        return new_solution

    def select_solution(self, population, bee_to_select):
        selected_bee = {}
        for key in population.keys():
            selected_bee[key] = population[key][bee_to_select]
        return selected_bee

    def replace_solution(self, population, bee_to_replace, new_solution):
        for key in population.keys():
            population[key][bee_to_replace] = new_solution[key]
