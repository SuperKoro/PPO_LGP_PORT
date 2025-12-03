"""
Portfolio data structures for PPO+LGP.
Defines Gene and ActionIndividual classes (without GA evolution logic).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import random


@dataclass
class Gene:
    """
    Gene for one rule:
    - kind: "DR" or "MH"
    - name: "EDD", "SA", "GA", "PSO", etc.
    - w_raw: Raw weight (used for MH)
    """
    kind: str
    name: str
    w_raw: float = 1.0


@dataclass
class ActionIndividual:
    """
    Action individual (portfolio):
    - genes[0]: DR gene
    - genes[1..]: MH genes
    """
    genes: List[Gene]

    def __post_init__(self):
        if len(self.genes) < 2:
            raise ValueError("ActionIndividual must have at least 1 DR + 1 MH.")
        if self.genes[0].kind.upper() != "DR":
            raise ValueError("Gene 0 must have kind='DR'.")

    @property
    def dr_gene(self) -> Gene:
        """Get dispatching rule gene."""
        return self.genes[0]

    @property
    def mh_genes(self) -> List[Gene]:
        """Get metaheuristic genes."""
        return self.genes[1:]


def individual_normalized_weights(individual: ActionIndividual) -> List[float]:
    """
    Return normalized weight vector (sum = 1) for MH genes.
    Used for logging/display only, doesn't change system logic.
    """
    mh_genes = individual.mh_genes
    raw_ws = [max(0.0, g.w_raw) for g in mh_genes]
    total = sum(raw_ws)
    if total <= 0.0:
        # If all weights are <= 0, distribute evenly for display
        return [1.0 / len(raw_ws)] * len(raw_ws)
    return [w / total for w in raw_ws]


def describe_individual(individual: ActionIndividual) -> str:
    """
    Return string description of portfolio:
    DR=EDD | SA(raw=3.40,norm=0.50) ; GA(raw=1.70,norm=0.25) ; PSO(raw=1.70,norm=0.25)
    """
    dr = individual.dr_gene
    mh_genes = individual.mh_genes
    norm_ws = individual_normalized_weights(individual)

    parts = []
    for g, w_norm in zip(mh_genes, norm_ws):
        parts.append(f"{g.name}(raw={g.w_raw:.2f}, norm={w_norm:.2f})")

    return f"DR={dr.name} | " + " ; ".join(parts)


class ActionLGP:
    """
    Simple LGP to INITIALIZE a pool of random ActionIndividuals.
    Pool evolution is handled in coevolution_trainer.
    """
    def __init__(self,
                 dr_list: Sequence[str],
                 mh_list: Sequence[str],
                 pool_size: int = 64,
                 n_mh_genes: int = 3,
                 seed: int | None = 0):
        if not dr_list:
            raise ValueError("Need at least 1 dispatching rule in dr_list.")
        if not mh_list:
            raise ValueError("Need at least 1 metaheuristic in mh_list.")

        self.rng = random.Random(seed)
        self.dr_list = [d.upper() for d in dr_list]
        self.mh_list = [m.upper() for m in mh_list]
        self.pool_size = int(pool_size)
        self.n_mh_genes = int(n_mh_genes)

        self.pool: List[ActionIndividual] = [
            self._random_individual()
            for _ in range(self.pool_size)
        ]

    def _random_individual(self) -> ActionIndividual:
        """Generate a random portfolio."""
        g0 = Gene(
            kind="DR",
            name=self.rng.choice(self.dr_list),
            w_raw=1.0
        )
        genes: List[Gene] = [g0]

        for _ in range(self.n_mh_genes):
            genes.append(
                Gene(
                    kind="MH",
                    name=self.rng.choice(self.mh_list),
                    w_raw=self.rng.uniform(0.1, 1.5)
                )
            )
        return ActionIndividual(genes=genes)
