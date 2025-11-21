# make_phase2_fd_beta_dataset.py
from __future__ import annotations

import os
from src.data_vector_fd_beta import sample_domain_grid_and_random_fd_beta


def main():
    cache_path = os.environ.get("PHASE2_FD_BETA_CACHE", "datasets/phase2_fd_beta_input_big.npz")

    n_train = int(os.environ.get("PHASE2_FD_BETA_NTRAIN", "100000"))
    n_val   = int(os.environ.get("PHASE2_FD_BETA_NVAL",   "20000"))

    Xtr, Ytr, Xva, Yva, meta = sample_domain_grid_and_random_fd_beta(
        n_train=n_train,
        n_val=n_val,
        cache_path=cache_path,
    )

    print("\n[FD-β Dataset] Done.")
    print(f"  train: X={Xtr.shape}, Y={Ytr.shape}")
    print(f"  val  : X={Xva.shape}, Y={Yva.shape}")
    print(f"  cache: {cache_path}")
    print(f"  meta : {meta}")


if __name__ == "__main__":
    main()
