import argparse
import os
import pickle
import random
import time

import numpy as np
import scipy.stats
import torch.cuda
import torch.utils.data

from deap import algorithms
from deap import creator
from deap import base
from deap import tools

from augmentation import COLLECTION, AugmentationSetting, get_augmentation_policy
from dataset import ImageDataset
from metrics import wasserstein_distance, variance
from networks import compute_features, get_backbone
from plotting import plot_logbook, plot_frontier

MIN_AUG_LENGTH = 2
MAX_AUG_LENGTH = 17

FIXED_AUG_LENGTH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataloader(dataset: ImageDataset, add_transforms=None, batch_size=16, num_workers=0, sample_size=None):
    if add_transforms is not None:
        dataset.transforms = add_transforms

    sample_size = len(dataset) if sample_size is None else sample_size

    dataset_subset = torch.utils.data.Subset(dataset, np.arange(0, sample_size))
    dataloader = torch.utils.data.DataLoader(dataset_subset, batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers, pin_memory=True,
                                                       collate_fn=dataset.collate_fn)
    return dataloader


def generate_augmentation_sequence(nested_size: int = 1):
    if nested_size == 1:
        return generate_augmentation()
    else:
        return [generate_augmentation() for _ in range(nested_size)]


def generate_augmentation():
    """
    Generate a random individual augmentation in the form of a tuple
    """
    aug = random.choice(list(COLLECTION.keys()))
    prob = random.uniform(0.0, 1.0)
    strength = random.uniform(0.0, 2.0)

    return (aug, prob, strength)


def random_augmentation_length():
    """
    Get a random length for an augmentation sequence
    Sampled from a truncated normal distribution
    """

    clip_a = MIN_AUG_LENGTH
    clip_b = MAX_AUG_LENGTH
    mean = 6
    std = 5

    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return int(scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=std))


def generate_individual(individual_class, nested_size: int, fixed_length: int = None):
    """
    Generate a random augmentation sequence with a random length
    """
    length = random_augmentation_length() if fixed_length is None else fixed_length
    return individual_class([generate_augmentation_sequence(nested_size) for _ in range(length)])


def mutate_individual(individual,  mutation_prob, nested_size: int):
    """
    Mutate an individual
    """
    for i in range(len(individual)):
        if random.uniform(0.0, 1.0) < mutation_prob:
            individual[i] = generate_augmentation_sequence(nested_size)

    if FIXED_AUG_LENGTH is None:
        if len(individual) < MAX_AUG_LENGTH and random.uniform(0.0, 1.0) < mutation_prob:
            individual.append(generate_augmentation_sequence(nested_size))

        if len(individual) > 2 and random.uniform(0.0, 1.0) < mutation_prob:
            del individual[random.randrange(0, len(individual))]

    return individual,


def individual_equal(a, b):
    """
    Determine if two individuals (augmentation policies) are the same
    """
    if len(a) != len(b):
        return False

    for aug_a, aug_b in zip(a, b):
        if aug_a[0] != aug_b[0]:
            return False
    return True


def create_augmentation_setting(individual: list, pick_n: int = None):
    if isinstance(individual[0], list):
        # in case of nested augmentations
        setting_dict = {
            "augmentation": [[a[0] for a in sequence] for sequence in individual],
            "augmentation_probability": [[a[1] for a in sequence] for sequence in individual],
            "augmentation_strength": [[a[2] for a in sequence] for sequence in individual],
            "augmentation_pick_n": pick_n
        }
    else:
        # in case of a flat augmentation list
        setting_dict = {
            "augmentation": [a[0] for a in individual],
            "augmentation_probability": [a[1] for a in individual],
            "augmentation_strength": [a[2] for a in individual],
            "augmentation_pick_n": pick_n
        }

    return AugmentationSetting(setting_dict)


def evaluate_individual(individual, reference_features, train_loader, network, pick_n: int = None):
    """
    Evaluate and individual of the generation by computing the distance and variation metrics
    """
    start = time.time()
    setting = create_augmentation_setting(individual, pick_n=pick_n)
    transform = get_augmentation_policy(setting)

    train_loader.dataset.dataset.transforms = transform

    train_features = compute_features(network, train_loader, device=DEVICE, transform=True)

    dist = wasserstein_distance(reference_features, train_features)
    var = variance(train_features)

    print(f"Eval individual\tDist: {dist:.2f}\tVar: {var:.2f}\tTime: {time.time() - start:.3f} s")

    return var, dist


def create_toolbox(nested_size: int, fixed_length: int):
    creator.create("Fitness", base.Fitness, weights=(1.0, -2.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("individual", generate_individual, creator.Individual, nested_size=nested_size, fixed_length=fixed_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def create_stats():
    f_stats = tools.Statistics(lambda ind: ind.fitness.values)
    f_stats.register("avg", np.mean, axis=0)
    f_stats.register("min", np.min, axis=0)
    f_stats.register("max", np.max, axis=0)

    s_stats = tools.Statistics(key=len)
    s_stats.register("avg", np.mean)
    s_stats.register("min", np.min)
    s_stats.register("max", np.max)

    total_stats = tools.MultiStatistics(fitness=f_stats, size=s_stats)

    return total_stats


def get_best_distance(individuals):
    total_fitness = [-i.fitness.wvalues[1] for i in individuals]
    indices = np.argsort(total_fitness)

    return individuals[indices[0]]

@torch.no_grad()
def learn_strategies(reference_folder: str, train_folder: str, generations: int, population_size: int, sample_size: int = 256, pick_n: int = None, nested_size: int = 1, fixed_length: int = None):
    dataset_reference = ImageDataset(reference_folder, in_memory=False)
    dataset_train = ImageDataset(train_folder, in_memory=False)

    reference_loader = create_dataloader(dataset_reference, sample_size=sample_size)
    train_loader = create_dataloader(dataset_train, sample_size=sample_size)

    backbone = get_backbone().to(DEVICE)

    print(f"Compute reference features")
    reference_features = compute_features(backbone, reference_loader, device=DEVICE, transform=True)

    toolbox = create_toolbox(nested_size=nested_size, fixed_length=fixed_length)

    toolbox.register("evaluate", evaluate_individual, reference_features=reference_features, train_loader=train_loader, network=backbone, pick_n=pick_n)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate_individual, mutation_prob=0.1, nested_size=nested_size)
    toolbox.register("select", tools.selNSGA2)

    CXPB = 0.6
    MUTPB = 0.3

    pop = toolbox.population(n=population_size)
    hof = tools.ParetoFront(individual_equal)
    total_stats = create_stats()

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, population_size, population_size * 2, CXPB, MUTPB, generations, total_stats, halloffame=hof)

    plot_logbook(logbook)

    to_save = get_best_distance(hof)

    plot_frontier(pop, hof, to_save)
    setting = create_augmentation_setting(to_save)

    print(f"Found Strategy:\n")
    print(setting)

    with open("augmentation.pkl", 'wb') as f:
        pickle.dump(setting, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", type=str, required=True)
    parser.add_argument("--reference-folder", type=str, required=True)

    parser.add_argument("--pick-n", type=int, default=None)
    parser.add_argument("--nested-size", type=int, default=1)
    parser.add_argument("--fixed-length", type=int, default=None)

    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=50)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    learn_strategies(
        reference_folder=args.reference_folder,
        train_folder=args.train_folder,
        generations=args.generations,
        population_size=args.population,
        pick_n=args.pick_n,
        nested_size=args.nested_size,
        fixed_length=args.fixed_length,
        sample_size=16
    )
