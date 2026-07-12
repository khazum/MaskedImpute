import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict_dropouts_new import biozero_posterior_from_nb_dropout


def test_biozero_posterior_uses_latent_zero_probability_as_numerator():
    p0_nb = np.array([0.05, 0.25, 0.80], dtype=float)
    p_drop = np.array([0.10, 0.40, 0.75], dtype=float)

    expected = p0_nb / (p_drop + (1.0 - p_drop) * p0_nb)
    observed = biozero_posterior_from_nb_dropout(p0_nb, p_drop)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_biozero_posterior_is_not_dropout_survival_weighted():
    p0_nb = np.array([0.25], dtype=float)
    p_drop = np.array([0.40], dtype=float)

    corrected = biozero_posterior_from_nb_dropout(p0_nb, p_drop)
    old_survival_weighted = (1.0 - p_drop) * p0_nb / (
        p_drop + (1.0 - p_drop) * p0_nb
    )

    assert corrected[0] > old_survival_weighted[0]
    np.testing.assert_allclose(corrected[0], 0.25 / (0.40 + 0.60 * 0.25))


if __name__ == "__main__":
    test_biozero_posterior_uses_latent_zero_probability_as_numerator()
    test_biozero_posterior_is_not_dropout_survival_weighted()
    print("biozero prior sanity checks passed")
