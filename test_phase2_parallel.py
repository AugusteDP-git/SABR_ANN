import numpy as np
from src.data_vector_true import sample_domain_grid_and_random


def main():
    X_tr, Y_tr, X_val, Y_val = sample_domain_grid_and_random(
        n_random_train=5000, n_val=2000, include_grid=False
    )
    print("train", X_tr.shape, Y_tr.shape, "val", X_val.shape, Y_val.shape)
    print("vol range (train):", float(Y_tr.min()), "->", float(Y_tr.max()))
    if np.isfinite(Y_tr).all():
        print("all finite vols")
    else:
        bad = np.argwhere(~np.isfinite(Y_tr))
        print("non-finite at indices:", bad[:5])


if __name__ == "__main__":
    main()
