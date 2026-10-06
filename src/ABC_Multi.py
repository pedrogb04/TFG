import numpy as np
import time
import pandas as pd
import sys
import warnings

from numpy.linalg import LinAlgError
from src.ABC_Base import ABC_Base


class ABC_Multi(ABC_Base):
    def __init__(self, objective_function, r_dim, d_dim, boundaries, limit):
        super().__init__(objective_function, r_dim, d_dim, boundaries, limit)

        self.ideal_point = np.full(2, np.inf)
        self.worst_point = np.full(2, -np.inf)
        self.extreme_points = None

    def get_initial_population(self, population_size):
        initial_population = {
            "real": np.zeros((4 * population_size, self.r_dim), dtype=float),
            "discrete": np.zeros((4 * population_size, self.d_dim), dtype=int),
            "accuracy": np.zeros(4 * population_size, dtype=float),
            "num_features": np.zeros(4 * population_size, dtype=float),
            "trials": np.zeros(4 * population_size, dtype=int),
            "fitness": np.zeros(4 * population_size, dtype=float),
            "ranking": np.zeros(4 * population_size, dtype=int),
            "niche_count": np.zeros(4 * population_size, dtype=int),
        }

        employed_population = super().get_initial_population(population_size)
        employed_population["ranking"] = np.zeros(population_size, dtype=int)
        employed_population["niche_count"] = np.zeros(population_size, dtype=int)

        for i in range(population_size):
            self.replace_solution(
                initial_population, i, self.select_solution(employed_population, i)
            )

        return initial_population

    def perturb_solution(self, population, population_size, bee_to_perturb):
        new_solution = super().perturb_solution(
            population, population_size, bee_to_perturb
        )
        new_solution["ranking"] = 0
        new_solution["niche_count"] = 0
        return new_solution

    def employed_mo(self, population, population_size):
        for e in range(population_size):
            new_sol = self.perturb_solution(population, population_size, e)

            original_sol = self.select_solution(population, e)
            if self.dominate(original_sol, new_sol) == -1:
                self.replace_solution(population, e, new_sol)
                self.replace_solution(population, population_size + e, original_sol)
                population["trials"][population_size + e] = 0
            else:
                self.replace_solution(population, population_size + e, new_sol)
                population["trials"][e] += 1

    def onlookers_mo(self, population, population_size):
        employed = {}
        for key in ["ranking", "niche_count", "accuracy", "num_features"]:
            employed[key] = population[key][:population_size]
        self.calc_ranking_and_niche_count(employed, population_size)

        max_count = np.max(employed["niche_count"])

        fitness_mo = 1 / (employed["ranking"] + employed["niche_count"] / max_count)
        probs = 0.1 + 0.9 * fitness_mo

        e = 0
        o = 0

        while o < population_size:
            if np.random.rand() < probs[e]:
                new_sol = self.perturb_solution(population, population_size, e)

                self.replace_solution(population, 2 * population_size + o, new_sol)

                original_sol = self.select_solution(employed, e)
                if self.dominate(original_sol, new_sol) == -1:
                    population["trials"][e] = 0
                else:
                    population["trials"][e] += 1

                o += 1
            e = (e + 1) % population_size

    def scout_mo(self, population, population_size):
        replaced_sols = 0
        for i in range(population_size):
            if population["trials"][i] >= self.limit:
                scouted_sol = self.select_solution(population, i)
                self.replace_solution(
                    population, 3 * population_size + replaced_sols, scouted_sol
                )
                population["trials"][3 * population_size + replaced_sols] = 0

                new_pop = self.get_initial_population(1)
                new_sol = self.select_solution(new_pop, 0)

                self.replace_solution(population, i, new_sol)
                replaced_sols += 1

        return replaced_sols

    def optimize_mo(self, population_size, generations):
        population = self.get_initial_population(population_size)

        best_bee = {}
        best_solution_history = []
        convergence_curve = np.zeros(generations)

        print("ABC_MO is optimizing...")
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
            self.employed_mo(population, population_size)

            self.onlookers_mo(population, population_size)

            replaced_bees = self.scout_mo(population, population_size)

            util_population = 3 * population_size + replaced_bees
            final_population = {}
            for key in population.keys():
                final_population[key] = population[key][:util_population]

            self.sort_population(final_population, population_size)

            for key in population.keys():
                population[key][:population_size] = final_population[key][
                    :population_size
                ]

            best_fitness = np.argmax(population["fitness"][:population_size])
            best_bee = self.select_solution(population, best_fitness)
            best_solution_history.append(best_bee)
            convergence_curve[current_gen] = best_bee["fitness"]

            history.loc[len(history)] = [
                current_gen,
                best_bee["fitness"],
                (1.0 - best_bee["accuracy"]) * 100,
                time.perf_counter() - timer_start,
                best_bee["discrete"],
                best_bee["real"],
            ]

        print(f"The best bee has a fitness of: {best_bee["fitness"]}")

        self.solution_history = best_solution_history

        return history

    def sort_population(self, population, population_size):
        self.calc_ranking_and_niche_count(population, population_size)

        sorted_indices = np.lexsort((population["niche_count"], population["ranking"]))

        for key in population.keys():
            population[key] = population[key][sorted_indices]

    def calc_ranking_and_niche_count(self, population, population_size):
        F = np.column_stack((population["accuracy"], population["num_features"]))
        fronts = self.fast_non_dominated_sort(F)
        ranks = self.rank_from_fronts(fronts, F.shape[0])
        non_dominated = fronts[0]

        ref_dirs = self.das_dennis(n_partitions=population_size - 1, n_dim=2)

        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        self.extreme_points = self.get_extreme_points_c(
            F[non_dominated, :],
            self.ideal_point,
            extreme_points=self.extreme_points,
        )

        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        nadir_point = self.get_nadir_point(
            self.extreme_points,
            self.ideal_point,
            self.worst_point,
            worst_of_front,
            worst_of_population,
        )

        niche_of_individuals, _, _ = self.associate_to_niches(
            F, ref_dirs, self.ideal_point, nadir_point
        )

        niche_count = self.calc_niche_count(
            len(ref_dirs), niche_of_individuals=niche_of_individuals
        )

        population["ranking"] = ranks
        population["niche_count"] = niche_count[niche_of_individuals]

    # Calcular matriz de dominación
    def calc_domination_matrix(self, F, _F=None, epsilon=0.0):

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = (
            np.logical_and(smaller, np.logical_not(larger)) * 1
            + np.logical_and(larger, np.logical_not(smaller)) * -1
        )

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M

    # Calcular los frentes de Pareto
    def fast_non_dominated_sort(self, F, **kwargs):
        M = self.calc_domination_matrix(F)

        # calculate the dominance matrix
        n = M.shape[0]

        fronts = []

        if n == 0:
            return fronts

        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=int)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):

            for j in range(i + 1, n):
                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # append the first front to the current front
        fronts.append(current_front)

        # while not all solutions are assigned to a pareto front
        while n_ranked < n:

            next_front = []

            # for each individual in the current front
            for i in current_front:

                # all solutions that are dominated by this individuals
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts

    def rank_from_fronts(self, fronts, n):
        # create the rank array and set values
        rank = np.full(n, sys.maxsize, dtype=int)
        for i, front in enumerate(fronts):
            rank[front] = i + 1

        return rank

    def associate_to_niches(
        self, F, niches, ideal_point, nadir_point, utopian_epsilon=0.0
    ):
        utopian_point = ideal_point - utopian_epsilon

        denom = nadir_point - utopian_point
        denom[denom == 0] = 1e-12

        # normalize by ideal point and intercepts
        N = (F - utopian_point) / denom
        dist_matrix = self.calc_perpendicular_distance(N, niches)

        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

        return niche_of_individuals, dist_to_niche, dist_matrix

    def calc_perpendicular_distance(self, N, ref_dirs):
        """Calculate perpendicular distance from points to reference directions."""
        u = np.tile(ref_dirs, (len(N), 1))
        v = np.repeat(N, len(ref_dirs), axis=0)

        norm_u = np.linalg.norm(u, axis=1)

        scalar_proj = np.sum(v * u, axis=1) / norm_u
        proj = scalar_proj[:, None] * u / norm_u[:, None]
        val = np.linalg.norm(proj - v, axis=1)
        matrix = np.reshape(val, (len(N), len(ref_dirs)))

        return matrix

    def calc_niche_count(self, n_niches, niche_of_individuals):
        niche_count = np.zeros(n_niches, dtype=int)
        index, count = np.unique(niche_of_individuals, return_counts=True)
        niche_count[index] = count
        return niche_count

    def das_dennis(self, n_partitions, n_dim):
        if n_partitions == 0:
            return np.full((1, n_dim), 1 / n_dim)
        else:
            ref_dirs = []
            ref_dir = np.full(n_dim, np.nan)
            self.das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
            return np.concatenate(ref_dirs, axis=0)

    def das_dennis_recursion(self, ref_dirs, ref_dir, n_partitions, beta, depth):
        if depth == len(ref_dir) - 1:
            ref_dir[depth] = beta / (1.0 * n_partitions)
            ref_dirs.append(ref_dir[None, :])
        else:
            for i in range(beta + 1):
                ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                self.das_dennis_recursion(
                    ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1
                )

    def get_extreme_points_c(self, F, ideal_point, extreme_points=None):
        # calculate the asf which is used for the extreme point decomposition
        weights = np.eye(F.shape[1])
        weights[weights == 0] = 1e6

        # add the old extreme points to never lose them for normalization
        _F = F
        if extreme_points is not None:
            _F = np.concatenate([extreme_points, _F], axis=0)

        # use __F because we substitute small values to be 0
        __F = _F - ideal_point
        __F[__F < 1e-3] = 0

        # update the extreme points for the normalization having the highest asf value each
        F_asf = np.max(__F * weights[:, None, :], axis=2)

        I = np.argmin(F_asf, axis=1)
        extreme_points = _F[I, :]

        return extreme_points

    def get_nadir_point(
        self,
        extreme_points,
        ideal_point,
        worst_point,
        worst_of_front,
        worst_of_population,
    ):
        try:

            # find the intercepts using gaussian elimination
            M = extreme_points - ideal_point
            b = np.ones(extreme_points.shape[1])
            plane = np.linalg.solve(M, b)

            warnings.simplefilter("ignore")
            intercepts = 1 / plane

            nadir_point = ideal_point + intercepts

            # check if the hyperplane makes sense
            if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
                raise LinAlgError()

            # if the nadir point should be larger than any value discovered so far set it to that value
            # NOTE: different to the proposed version in the paper
            b = nadir_point > worst_point
            nadir_point[b] = worst_point[b]

        except LinAlgError:

            # fall back to worst of front otherwise
            nadir_point = worst_of_front

        # if the range is too small set it to worst of population
        b = nadir_point - ideal_point <= 1e-6
        nadir_point[b] = worst_of_population[b]

        return nadir_point

    def dominate(self, solution_a, solution_b):
        # Returns 1 if a dominates b, -1 if b dominates a and 0 if neither dominates the other
        flag_a = flag_b = 0
        accuracy_a = solution_a["accuracy"]
        accuracy_b = solution_b["accuracy"]
        features_a = solution_a["num_features"]
        features_b = solution_b["num_features"]
        if accuracy_a < accuracy_b:
            flag_a = 1
        else:
            if accuracy_b < accuracy_a:
                flag_b = 1
        if features_a < features_b:
            flag_a = 1
        else:
            if features_b < features_a:
                flag_b = 1

        if flag_a == 1 and flag_b == 0:
            return 1
        if flag_a == 0 and flag_b == 1:
            return -1

        return 0
