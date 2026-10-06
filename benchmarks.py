"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm."
           Information sciences 179.13 (2009): 2232-2248.

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

 -- Purpose: Defining the benchmark function for the Gravitational Search Algorithm (GSA)

Code compatible:
 -- Python >= 3.9
"""

import numpy as np
import pandas as pd

from typing import Any, Mapping, Tuple


def F1(x: Mapping[str, np.ndarray]) -> np.signedinteger[Any]:
    """
    Spere Function

    Args:
      x: numpy.ndarray : input vector of decision variables

    Returns:
      float : output of the Sphere function
    """
    return np.sum(x["real"] ** 2)


def gsa_svm_fitness(
    accuracy: float,
    discrete_array: np.ndarray,
    wf: float = 0.2,
    wa: float = 0.8,
) -> Tuple[float, float]:
    """
    Fitness function for the GSA-SVM algorithm

    Args:
        accuracy (float): accuracy
        solution (Mapping[str, np.ndarray]): solution vector
        wf (float): weight for the fitness function
        wa (float): weight for the accuracy

    Returns:
        Tuple[float, float]: fitness and accuracy
    """

    features = 1 - sum(discrete_array) / len(discrete_array)
    fitness = accuracy * wa + features * 100 * wf
    return fitness, features


def multiobjective_fitness(
    accuracy: float,
    discrete_array: np.ndarray,
) -> Tuple[float, int]:

    # Ambas deben minimizar
    num_features = sum(discrete_array) / len(discrete_array)
    minimized_accuracy = 1.0 - accuracy

    return num_features, minimized_accuracy


def get_function_details(a: int) -> Any:
    """
    Get the details of the benchmark function

    Args:
        a: int : index of the benchmark function

    Returns:
        Any : name, lower bound, upper bound, dimensions of the benchmark function
    """
    param = {0: {"function": F1, "lower_bound": -100, "upper_bound": 100, "dim": 10}}
    return param.get(a, "nothing")
