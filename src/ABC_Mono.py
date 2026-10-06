import numpy as np
import time
import pandas as pd
from src.ABC_Base import ABC_Base


class ABC_Mono(ABC_Base):
    def __init__(self, objective_function, r_dim, d_dim, boundaries, limit):
        super().__init__(objective_function, r_dim, d_dim, boundaries, limit)

    def employed_bees_phase(self, population, population_size):
        for e in range(population_size):
            new_sol = self.perturb_solution(population, population_size, e)

            w = np.argmin(population["fitness"])
            if new_sol["fitness"] > population["fitness"][w]:
                self.replace_solution(population, w, new_sol)
                population["trials"][e] = 0
            else:
                population["trials"][e] += 1

    def onlooker_bees_phase(self, population, population_size):
        fitness_norm = population["fitness"] - np.min(population["fitness"])
        sum_fit = np.sum(fitness_norm)
        probs = (
            0.1 + 0.9 * (fitness_norm / sum_fit)
            if sum_fit != 0
            else np.ones(population_size) / population_size
        )

        o = 0
        e = 0
        while o < population_size:
            if np.random.rand() < probs[e]:
                o += 1

                new_sol = self.perturb_solution(population, population_size, e)

                w = np.argmin(population["fitness"])

                if new_sol["fitness"] > population["fitness"][w]:
                    self.replace_solution(population, w, new_sol)
                    population["trials"][e] = 0
                else:
                    population["trials"][e] += 1
            e = (e + 1) % population_size

    def scout_bees_phase(self, population, population_size):
        for s in range(population_size):
            if population["trials"][s] >= self.limit:
                new_pop = self.get_initial_population(1)
                new_sol = self.select_solution(new_pop, 0)
                self.replace_solution(population, s, new_sol)

    def optimize(self, population_size, generations):
        population = self.get_initial_population(population_size)

        best_bee = {}
        best_global_fitness = 0.0
        best_solution_history = []
        convergence_curve = np.zeros(generations)

        print("ABC is optimizing...")
        timer_start = time.perf_counter()

        history = pd.DataFrame(
            columns=[
                "Generation",
                "Fitness",
                "Accuracy",
                "ExecutionTime",
                "Discrete",
                "Real",
            ]
        )

        for current_gen in range(generations):

            self.employed_bees_phase(population, population_size)

            self.onlooker_bees_phase(population, population_size)

            self.scout_bees_phase(population, population_size)

            # Update of the best global
            best_fitness = np.argmax(population["fitness"])
            if population["fitness"][best_fitness] > best_global_fitness:
                best_global_fitness = population["fitness"][best_fitness]
                best_bee = self.select_solution(population, best_fitness)

            convergence_curve[current_gen] = best_global_fitness
            best_solution_history.append(best_bee)
            history.loc[len(history)] = [
                current_gen,
                best_global_fitness,
                best_bee["accuracy"],
                time.perf_counter() - timer_start,
                best_bee["discrete"],
                best_bee["real"],
            ]

            print(
                f"At iteration {current_gen + 1} the best fitness is {best_global_fitness}"
            )

        self.convergence = convergence_curve
        self.solution_history = best_solution_history

        return history
