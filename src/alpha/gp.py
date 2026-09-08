"""A small, reproducible genetic-programming engine for formula dry runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formula import Expression


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 120
    generations: int = 12
    max_depth: int = 3
    tournament_size: int = 4
    elite_count: int = 12
    crossover_rate: float = 0.65
    mutation_rate: float = 0.25
    seed: int = 7
    seed_all_terminals: bool = False


def _paths(expression: Expression, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [path]
    for index, child in enumerate(expression.children):
        paths.extend(_paths(child, path + (index,)))
    return paths


def _at(expression: Expression, path: tuple[int, ...]) -> Expression:
    node = expression
    for index in path:
        node = node.children[index]
    return node


def _replace(expression: Expression, path: tuple[int, ...], replacement: Expression) -> Expression:
    if not path:
        return replacement
    index = path[0]
    children = list(expression.children)
    children[index] = _replace(children[index], path[1:], replacement)
    return Expression(expression.op, tuple(children), expression.name)


def _random_tree(names: tuple[str, ...], max_depth: int, rng: np.random.Generator, depth: int = 1) -> Expression:
    if depth >= max_depth or rng.random() < 0.35:
        return Expression("primitive", name=str(rng.choice(names)))
    if rng.random() < 0.35:
        return Expression(str(rng.choice(("neg", "abs"))), (_random_tree(names, max_depth, rng, depth + 1),))
    return Expression(
        str(rng.choice(("add", "sub", "mul", "div"))),
        (_random_tree(names, max_depth, rng, depth + 1), _random_tree(names, max_depth, rng, depth + 1)),
    )


def _tournament(population: list[Expression], fitness: dict[str, float], size: int, rng: np.random.Generator) -> Expression:
    indices = rng.integers(0, len(population), size=size)
    return max((population[int(index)] for index in indices), key=lambda item: fitness[item.to_string()])


def evolve_formulae(names, score, config: GPConfig = GPConfig()):
    """Evolve protected expression trees with a caller-supplied fitness score.

    ``score`` must be deterministic and use only the discovery period.  It is
    called with an ``Expression`` and should return a larger-is-better scalar.
    """
    if config.population_size <= config.elite_count or config.max_depth < 1:
        raise ValueError("invalid GP population or depth configuration")
    terminals = tuple(names)
    if not terminals:
        raise ValueError("at least one terminal is required")
    rng = np.random.default_rng(config.seed)
    if config.seed_all_terminals:
        if config.population_size < len(terminals):
            raise ValueError("population_size must cover every terminal when seed_all_terminals is enabled")
        population = [Expression("primitive", name=name) for name in terminals]
        population += [_random_tree(terminals, config.max_depth, rng)
                       for _ in range(config.population_size - len(population))]
    else:
        population = [_random_tree(terminals, config.max_depth, rng) for _ in range(config.population_size)]
    cache: dict[str, float] = {}
    history = []
    for generation in range(config.generations):
        for expression in population:
            name = expression.to_string()
            if name not in cache:
                value = float(score(expression))
                cache[name] = value if np.isfinite(value) else -np.inf
        population = sorted(population, key=lambda item: cache[item.to_string()], reverse=True)
        history.append({"generation": generation, "best_formula": population[0].to_string(), "best_fitness": cache[population[0].to_string()]})
        next_population = population[:config.elite_count]
        while len(next_population) < config.population_size:
            parent = _tournament(population, cache, config.tournament_size, rng)
            draw = rng.random()
            if draw < config.crossover_rate:
                donor = _tournament(population, cache, config.tournament_size, rng)
                parent_paths, donor_paths = _paths(parent), _paths(donor)
                parent_path = parent_paths[int(rng.integers(len(parent_paths)))]
                donor_path = donor_paths[int(rng.integers(len(donor_paths)))]
                child = _replace(parent, parent_path, _at(donor, donor_path))
            elif draw < config.crossover_rate + config.mutation_rate:
                parent_paths = _paths(parent)
                parent_path = parent_paths[int(rng.integers(len(parent_paths)))]
                child = _replace(parent, parent_path, _random_tree(terminals, config.max_depth, rng))
            else:
                child = parent
            if child.depth <= config.max_depth:
                next_population.append(child)
        population = next_population
    ranked = sorted(cache, key=cache.get, reverse=True)
    return {"ranked": ranked, "fitness": cache, "history": history}
