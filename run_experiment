"""
run_experiment.py

Run DTMKFC on a selected dataset with optional constraints.
"""

import argparse
import numpy as np
from dtmkfc import DTMKFC
from utils import evaluate_clustering, sample_constraints
from datasets import get_dataset


def main(args):
    print(f"[run] loading dataset '{args.dataset}' …")
    views, labels = get_dataset(args.dataset)

    if args.constraints > 0:
        constraints = sample_constraints(labels,
                                         num_constraints=args.constraints,
                                         ratio=args.ratio)
    else:
        constraints = None

    print("[run] initializing model …")
    model = DTMKFC(n_clusters=len(np.unique(labels)),
                   n_neighbors=args.k,
                   gamma=args.gamma,
                   n_iter=args.n_iter,
                   alpha=args.alpha,
                   beta=args.beta)

    print("[run] training …")
    model.fit(views, constraints)
    pred = model.predict()

    print("[run] evaluation:")
    evaluate_clustering(labels, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="Dataset name: synthetic | orl | scene15 | msrcv1 | nuswide | leaves")
    parser.add_argument("--constraints", type=int, default=100,
                        help="Number of pairwise constraints (0 = unsupervised)")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Must-link ratio among all constraints")
    parser.add_argument("--k", type=int, default=10,
                        help="kNN neighbors")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="RBF gamma")
    parser.add_argument("--n_iter", type=int, default=10,
                        help="Graph refinement steps")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Constraint influence")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Topology feedback weight")
    args = parser.parse_args()
    main(args)
