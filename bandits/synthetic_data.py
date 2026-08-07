import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


class AWMGenerator:
    """
    Context generator with A in {-1, 1}, each with probability 0.5.
    W and M remain binary in {0, 1}.
    """
    def __init__(
        self,
        seed=None,
        beta_w0=0.0,
        beta_wa=1.20,
        beta_m0=0.0,
        beta_ma=1.00,
        beta_mw=1.00,
    ):
        self.rng = np.random.default_rng(seed)

        self.beta_w0 = float(beta_w0)
        self.beta_wa = float(beta_wa)

        self.beta_m0 = float(beta_m0)
        self.beta_ma = float(beta_ma)
        self.beta_mw = float(beta_mw)

        # A is now in {-1, 1}; W,M stay in {0,1}
        self.context_support = [
            (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1),
            ( 1, 0, 0), ( 1, 0, 1), ( 1, 1, 0), ( 1, 1, 1),
        ]

        self._block = []
        self._block_ptr = 0
        self._refresh_block()

    @staticmethod
    def _bern_prob(p, x):
        return float(p) if int(x) == 1 else float(1.0 - p)

    def p_a(self, a):
        a = int(a)
        if a not in (-1, 1):
            raise ValueError(f"a must be -1 or 1, got {a}")
        return 0.5

    def p_w_given_a(self, a):
        a = int(a)
        if a not in (-1, 1):
            raise ValueError(f"a must be -1 or 1, got {a}")
        return float(sigmoid(self.beta_w0 + self.beta_wa * a))

    def p_m_given_aw(self, a, w):
        a = int(a)
        w = int(w)
        if a not in (-1, 1):
            raise ValueError(f"a must be -1 or 1, got {a}")
        if w not in (0, 1):
            raise ValueError(f"w must be 0 or 1, got {w}")

        return float(
            sigmoid(
                self.beta_m0
                + self.beta_ma * a
                + self.beta_mw * (2 * w - 1)
            )
        )

    def context_pmf(self):
        pmf = {}
        for a, w, m in self.context_support:
            p = (
                self.p_a(a)
                * self._bern_prob(self.p_w_given_a(a), w)
                * self._bern_prob(self.p_m_given_aw(a, w), m)
            )
            pmf[(a, w, m)] = float(p)
        return pmf

    def balanced_context_pmf(self):
        mass = 1.0 / len(self.context_support)
        return {ctx: mass for ctx in self.context_support}

    def pmf_W_given_A(self, a):
        p1 = self.p_w_given_a(a)
        return np.array([1.0 - p1, p1], dtype=float)

    def pmf_M_given_AW(self, a, w):
        p1 = self.p_m_given_aw(a, w)
        return np.array([1.0 - p1, p1], dtype=float)

    def all_probs_for_at(self, a_t):
        a_t = int(a_t)
        if a_t not in (-1, 1):
            raise ValueError(f"a_t must be -1 or 1, got {a_t}")

        a_alt = -a_t

        pW_a = self.pmf_W_given_A(a_t)
        pW_alt = self.pmf_W_given_A(a_alt)

        rows = []
        for w in (0, 1):
            pM_a = self.pmf_M_given_AW(a_t, w)
            pM_alt = self.pmf_M_given_AW(a_alt, w)

            for m in (0, 1):
                rows.append(
                    (
                        int(w),
                        int(m),
                        float(pW_a[w]),
                        float(pW_alt[w]),
                        float(pM_a[m]),
                        float(pM_alt[m]),
                    )
                )
        return rows

    def _refresh_block(self):
        perm = self.rng.permutation(len(self.context_support))
        self._block = [self.context_support[i] for i in perm]
        self._block_ptr = 0

    def sample_context_balanced(self):
        if self._block_ptr >= len(self._block):
            self._refresh_block()

        a, w, m = self._block[self._block_ptr]
        self._block_ptr += 1
        wms = self.all_probs_for_at(a)
        return int(a), int(w), int(m), wms
