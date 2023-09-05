import bz2
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import dimod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dimod import SampleSet, concatenate
from dimod.vartypes import Vartype
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from tqdm import tqdm

from .stats import AbstractDistribution


def onedown(x: List[int], idx: int, vartype: Vartype):
    if idx < 0 or idx >= len(x):
        raise IndexError(
            f"Index must be between 0 and {len(x)-1}, but actual was {idx}"
        )
    if vartype == Vartype.BINARY:
        return np.array([-a + 1 if i == idx else a for i, a in enumerate(x)])
    elif vartype == Vartype.SPIN:
        return np.array([-a if i == idx else a for i, a in enumerate(x)])
    else:
        raise NotImplementedError


def get_all_flipped(X: List[List[int]], dim: int, vartype: Vartype) -> List[List[int]]:
    X_set = set()
    if len(X.shape) != 2:
        raise ValueError
    for x in X:
        X_set.add(tuple(x))
        for i in range(dim):
            X_set.add(tuple(onedown(x, i, vartype=vartype)))
    return np.array([np.array(x) for x in X_set])


def get_stein_score(x: List[int], distrib: AbstractDistribution, vartype: Vartype):
    """
    difference score function
    Defined by (1) of Yang et al. 2018; Goodness-of-Fit Testing for Discrete Distributions via Stein Discrepancy
    """
    return np.array(
        list(
            map(
                lambda i: 1 - distrib.pmf(onedown(x, i, vartype)) / distrib.pmf(x),
                np.arange(len(x)),
            )
        )
    )


def read_from_file(path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    # load interaction strength
    # J[i] denotes an interaction strength between source[i] and target[i]
    path_J = os.path.join(path, "J.csv")
    if os.path.exists(path_J):
        df = pd.read_csv(os.path.join(path, "J.csv"))
    else:
        raise FileNotFoundError(f"{path_J} not found")
    source, target, list_J = df["s"].to_numpy(), df["t"].to_numpy(), df["J"].to_numpy()
    assert len(source) == len(target) and len(source) == len(list_J)
    dict_J = {}
    for i in range(len(source)):
        s, t = source[i], target[i]
        dict_J[(s, t)] = list_J[i]
    # load local field terms
    # h[i] denotes a local field value of node[i]
    path_h = os.path.join(path, "h.csv")
    dict_h = {}
    if os.path.exists(path_h):
        df = pd.read_csv(os.path.join(path, "h.csv"))
        n, list_h = df["n"].to_numpy(), df["h"].to_numpy()
        assert len(n) == len(list_h)
        for i in range(len(n)):
            dict_h[n[i]] = list_h[i]
    return dict_J, dict_h


def create_dwave_sampler(endpoint: str, token: int, solver: str) -> EmbeddingComposite:
    return EmbeddingComposite(
        DWaveSampler(endpoint=endpoint, token=token, solver=solver)
    )


def gauge_transformation(
    h: Dict[int, int], J: Dict[Tuple[int, int], int]
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """
    Perform a gauge transformation to mitigate the effects of bias.
    A randomly generated binary vector is used to redefine the Ising parameters.
    This transformation oreserves the energy eigenvalues of H(s).
    """
    transformed_J, transformed_h, random_vector = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    for e in J.keys():
        n, m = e[0], e[1]
        if n not in random_vector.keys():
            random_vector[n] = np.random.choice([-1, 1])
        if m not in random_vector.keys():
            random_vector[m] = np.random.choice([-1, 1])
        a, b = random_vector[n], random_vector[m]
        transformed_J[e] = a * b * J[e]

    for n in h.keys():
        transformed_h[n] = random_vector[n] * h[n]

    return h, J


def scale_J_h(J, h, alpha_in=0.2):
    J_new, h_new = {}, {}
    for k in J.keys():
        J_new[k] = J[k] * alpha_in
    for k in h.keys():
        h_new[k] = h[k] * alpha_in
    return J_new, h_new


def flatten_sampleset(sampleset: SampleSet) -> List[List[int]]:
    X, freq = sampleset.record["sample"], sampleset.record["num_occurrences"]
    X_flatten = []
    for i, x in enumerate(X):
        for _ in range(freq[i]):
            X_flatten.append(x)
    return X_flatten


def dwave_sampling(
    sampler: EmbeddingComposite or FixedEmbeddingComposite,
    h: defaultdict,
    J: defaultdict,
    *,
    annealing_time=125,
    iter_num=10000,
    gauge_interval=100,
    show_progress_bar=False,
) -> dimod.SampleSet:
    samplesets = list()
    iterations = range(iter_num)
    if show_progress_bar is True:
        iterations = tqdm(range(iter_num))
    for _ in iterations:
        sampleset = sampler.sample_ising(
            h=h,
            J=J,
            num_reads=gauge_interval,
            annealing_time=annealing_time,
            auto_scale=False,
        )
        samplesets.append(sampleset)
        h, J = gauge_transformation(h, J)
        time.sleep(0.03)
    return concatenate(samplesets).aggregate()


def save_energy_hist_img(
    energy_list: List[float], selected_energy_list: List[float], file_name: str
):
    hist, selected_hist = defaultdict(int), defaultdict(int)
    for e in energy_list:
        hist[e] += 1
    for e in selected_energy_list:
        selected_hist[e] += 1
    enes = sorted(list(hist.keys()))  # All energies appeared in initial population
    ene_freq_norm = [hist[k] for k in enes]
    selected_ene_freq_norm = [selected_hist[k] for k in enes]
    width = 0.35
    fig, ax = plt.subplots()
    x = np.arange(len(enes))
    _ = ax.bar(x - width / 2, ene_freq_norm, width, label="Initial population")
    _ = ax.bar(
        x + width / 2, selected_ene_freq_norm, width, label="Selected population"
    )

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Normalized frequency grouped by the function value")
    ax.set_xlabel("Energy")
    ax.set_ylim(0, len(energy_list))
    ax.set_title("Distribution of black-box function value")
    ax.set_xticks(x, enes)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig(file_name)


def history_filter(history_ref, f):
    history = []
    for h in history_ref:
        if len(history) == 0:
            history.append(h)
        else:
            t = f(f(history), h)
            history.append(t)
    return history


def binary2spin(x):
    x = np.array(x)
    return 2 * x - 1


def spin2binary(x):
    x = np.array(x)
    return (x + 1) / 2


def pload(file):
    with bz2.open(file, "rb") as f:
        return pickle.load(f)


def pdump(file, obj):
    with bz2.open(file, "wb") as f:
        pickle.dump(obj, f)


class IsingSampler:
    @staticmethod
    def random(target: List[int], dim: int, num_samples: int) -> Any:
        return np.array([np.random.choice(target, dim) for _ in range(num_samples)])
