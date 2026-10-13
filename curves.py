"""US Treasury par yields from FRED, and the curves fitted to them."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# Label and FRED series id for each Treasury maturity.
US_TREASURY = [
    ("1M", "DGS1MO"),  # 1-Month
    ("3M", "DGS3MO"),  # 3-Month
    ("6M", "DGS6MO"),  # 6-Month
    ("1Y", "DGS1"),  # 1-Year
    ("2Y", "DGS2"),  # 2-Year
    ("3Y", "DGS3"),  # 3-Year
    ("5Y", "DGS5"),  # 5-Year
    ("7Y", "DGS7"),  # 7-Year
    ("10Y", "DGS10"),  # 10-Year
    ("20Y", "DGS20"),  # 20-Year
    ("30Y", "DGS30"),  # 30-Year
]

def load_treasury(api_key, observation_start="1900-01-01"):
    """FRED constant-maturity Treasury par yields (%), one column per tenor.

    Days with a missing quote are dropped. The index is the observation date.
    """
    fred = Fred(api_key=api_key)
    data = {
        label: fred.get_series(series_id, observation_start=observation_start)
        for label, series_id in US_TREASURY
    }
    frame = pd.DataFrame(data)
    frame.index.name = "Date"
    return frame.dropna()


class YieldCurve:
    """Continuously compounded zero, forward and discount curves built from one date of par yields."""

    MATURITY_YEARS = {
        "1M": 1 / 12,
        "3M": 3 / 12,
        "6M": 6 / 12,
        "1Y": 1,
        "2Y": 2,
        "3Y": 3,
        "5Y": 5,
        "7Y": 7,
        "10Y": 10,
        "20Y": 20,
        "30Y": 30,
    }

    METHODS = {
        "linear": "Linear",
        "cubic_spline": "Cubic spline",
        "nelson_siegel": "Nelson-Siegel",
        "nss": "Nelson-Siegel-Svensson",
        "gp": "Gaussian process",
        "no_arb_gp": "No-arbitrage Gaussian process",
    }

    def __init__(self, par_rates, method="linear"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}, got {method!r}")
        par_rates = pd.Series(par_rates, dtype=float).dropna()
        self.method = method
        self.name = self.METHODS[method]
        self.as_of = par_rates.name
        self.quotes = self._zero_curve_table(par_rates)
        self.maturities = self.quotes["maturity_years"].to_numpy()
        self.zeros = self.quotes["yield_cc_pct"].to_numpy()
        self._model_forward = None
        self._zero_fn = self._fit(self.maturities, self.zeros)

    @staticmethod
    def _par_to_continuous_zeros(par_pct, maturities):
        """Bootstrap semiannual Treasury par yields to continuously compounded zeros.

        par_pct are FRED constant-maturity par yields in percent. maturities are in
        years. The result is the continuous zero rate in percent, same order.
        Bills (up to 6 months) are treated as zeros. Longer par yields are
        bootstrapped on a semiannual coupon grid, with the par yield interpolated
        linearly between the quoted maturities.
        """
        par_pct = np.asarray(par_pct, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        order = np.argsort(maturities)
        tenors = maturities[order]
        coupons = par_pct[order] / 100.0

        n_periods = int(np.round(tenors[-1] * 2))
        grid = np.arange(1, n_periods + 1) / 2.0
        par_grid = np.interp(grid, tenors, coupons)

        discount = np.empty(n_periods)
        for n, coupon in enumerate(par_grid):
            if n == 0:
                discount[n] = 1.0 / (1.0 + coupon / 2.0)
            else:
                accrued = (coupon / 2.0) * discount[:n].sum()
                discount[n] = (1.0 - accrued) / (1.0 + coupon / 2.0)

        zeros = np.empty(len(tenors))
        for i, (tenor, coupon) in enumerate(zip(tenors, coupons)):
            if tenor <= 0.5:
                df = (1.0 + coupon / 2.0) ** (-2.0 * tenor)
            else:
                df = discount[int(np.round(tenor * 2)) - 1]
            zeros[i] = -np.log(df) / tenor * 100.0

        out = np.empty(len(tenors))
        out[order] = zeros
        return out

    @classmethod
    def _zero_curve_table(cls, par_rates):
        """One date of par yields (%) to maturities, continuous zeros (%) and discount factors."""
        table = par_rates.rename("par_rate_pct").to_frame()
        table["maturity_years"] = table.index.map(cls.MATURITY_YEARS)
        table = table.sort_values("maturity_years")
        # FRED quotes are semiannual par yields. Bootstrap them to continuous zeros.
        table["yield_cc_pct"] = cls._par_to_continuous_zeros(
            table["par_rate_pct"].to_numpy(),
            table["maturity_years"].to_numpy(),
        )
        table["discount_factor"] = np.exp(
            -table["yield_cc_pct"] / 100 * table["maturity_years"]
        )
        return table

    @staticmethod
    def _nelson_siegel(tau, beta0, beta1, beta2, lam):
        """Continuously compounded Nelson-Siegel yield at maturity tau (years)."""
        x = tau / lam
        exp_x = np.exp(-x)
        factor = np.where(np.abs(x) < 1e-8, 1.0, (1 - exp_x) / x)
        return beta0 + beta1 * factor + beta2 * (factor - exp_x)

    @classmethod
    def _fit_nelson_siegel(cls, tenors, yields):
        """Least-squares Nelson-Siegel parameters (beta0, beta1, beta2, lam)."""
        # beta0 ~ long rate, beta1 ~ short minus long, beta2 ~ hump, lam ~ decay in years
        p0 = [yields[-1], yields[0] - yields[-1], 0.0, 2.0]
        bounds = ([-5, -20, -20, 0.05], [20, 20, 20, 30])
        return curve_fit(cls._nelson_siegel, tenors, yields, p0=p0, bounds=bounds)[0]

    @staticmethod
    def _nelson_siegel_svensson(tau, beta0, beta1, beta2, beta3, lam1, lam2):
        """Continuously compounded Nelson-Siegel-Svensson yield at maturity tau (years)."""
        def loadings(lam):
            x = tau / lam
            exp_x = np.exp(-x)
            level = np.where(np.abs(x) < 1e-8, 1.0, (1 - exp_x) / x)
            return level, level - exp_x

        slope, curve1 = loadings(lam1)
        _, curve2 = loadings(lam2)
        return beta0 + beta1 * slope + beta2 * curve1 + beta3 * curve2

    @classmethod
    def _fit_nelson_siegel_svensson(cls, tenors, yields):
        """Least-squares Svensson parameters (beta0, beta1, beta2, beta3, lam1, lam2)."""
        # Six parameters, so try a few starts and keep the best fit that is not stuck on a bound.
        bounds = ([3, -10, -10, -10, 0.1, 1.0], [10, 10, 10, 10, 5, 30])
        starts = [
            [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 1.5, 8.0],
            [yields[-1], yields[0] - yields[-1], -1.0, -1.5, 0.5, 3.0],
            [yields[-1], yields[0] - yields[-1], -2.0, 2.0, 1.0, 10.0],
            [yields[-1], yields[0] - yields[-1], 2.0, -2.0, 2.0, 15.0],
        ]
        lower, upper = np.array(bounds[0]), np.array(bounds[1])
        best = None
        for start in starts:
            fitted_params = curve_fit(
                cls._nelson_siegel_svensson, tenors, yields, p0=start, bounds=bounds, maxfev=20000
            )[0]
            residual = cls._nelson_siegel_svensson(tenors, *fitted_params) - yields
            sse = np.sum(residual**2)
            on_bound = np.any(np.isclose(fitted_params, lower) | np.isclose(fitted_params, upper))
            if best is None or (on_bound, sse) < best[0]:
                best = ((on_bound, sse), fitted_params)
        return best[1]

    @staticmethod
    def _fit_gaussian_process(tenors, yields):
        """Gaussian process fitted to zero rates (%) by maturity (years)."""
        # Matérn kernel for a smooth curve, plus a little noise so the fit is not forced through every quote.
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=5.0, length_scale_bounds=(0.2, 50.0), nu=2.5
        ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.1))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=8,
            normalize_y=True,
            random_state=0,
        )
        return gp.fit(tenors.reshape(-1, 1), yields)

    @staticmethod
    def _softplus(x):
        """Positive smooth map, so the instantaneous forward cannot go negative."""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    @classmethod
    def _fit_no_arbitrage_gp(cls, tenors, yields, sigma=0.05):
        """Zero curve whose instantaneous forward is softplus of a Gaussian process.

        Returns the maturity grid, the zero rates and forwards on it (both in %),
        and the chosen length scale. sigma is the yield noise in percent.
        """
        # Latent curve g is a Gaussian process. The forward is softplus(g) > 0,
        # and the zero rate is the average of that forward up to each maturity.
        grid = np.unique(np.concatenate([np.linspace(0, tenors[-1], 61), tenors]))
        obs_idx = np.searchsorted(grid, tenors)

        def fit(length_scale):
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
            prior = kernel(grid.reshape(-1, 1))
            prior[np.diag_indices(len(grid))] += 1e-8
            weights_to_curve = np.linalg.cholesky(prior)

            def curve_from(weights):
                forward = cls._softplus(np.mean(yields) + weights_to_curve @ weights)
                gaps = np.diff(grid)
                integral = np.concatenate(
                    [[0.0], np.cumsum(0.5 * (forward[1:] + forward[:-1]) * gaps)]
                )
                zero = integral[1:] / grid[1:]
                return zero, forward

            def loss(weights):
                zero, _ = curve_from(weights)
                fitted = zero[obs_idx - 1]
                return np.sum((fitted - yields) ** 2) / sigma**2 + np.sum(weights**2)

            result = minimize(loss, np.zeros(len(grid)), method="L-BFGS-B")
            zero, forward = curve_from(result.x)
            return result.fun, zero, forward

        best = None
        for length_scale in (1.5, 2.0, 3.0, 5.0, 8.0):
            objective, zero, forward = fit(length_scale)
            if best is None or objective < best[0]:
                best = (objective, length_scale, zero, forward)

        _, length_scale, zero, forward = best
        return grid[1:], zero, forward, length_scale

    def _fit(self, t, y):
        """Return a function from maturities (array, years) to zero rates (%)."""
        if self.method == "linear":
            return lambda x: np.interp(x, t, y)
        if self.method == "cubic_spline":
            return CubicSpline(t, y)
        if self.method == "nelson_siegel":
            params = self._fit_nelson_siegel(t, y)
            return lambda x: self._nelson_siegel(x, *params)
        if self.method == "nss":
            params = self._fit_nelson_siegel_svensson(t, y)
            return lambda x: self._nelson_siegel_svensson(x, *params)
        if self.method == "gp":
            gp = self._fit_gaussian_process(t, y)
            return lambda x: gp.predict(x.reshape(-1, 1))
        grid, zero, forward, _ = self._fit_no_arbitrage_gp(t, y)
        forward_grid = np.concatenate([[0.0], grid])
        self._model_forward = lambda x: np.interp(x, forward_grid, forward)
        return lambda x: np.interp(x, grid, zero)

    @staticmethod
    def _evaluate(fn, t):
        """Apply fn to a scalar or array of maturities, returning the same shape."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.asarray(fn(t_arr), dtype=float).reshape(t_arr.shape)
        return out.item() if np.ndim(t) == 0 else out

    def zero(self, t):
        """Continuously compounded zero rate (%) at maturity t (years)."""
        return self._evaluate(self._zero_fn, t)

    def discount(self, t):
        """Discount factor D(t) = exp(-r(t) t)."""
        return np.exp(-np.asarray(self.zero(t)) / 100 * np.asarray(t, dtype=float))

    def forward(self, t):
        """Instantaneous forward rate (%) at maturity t (years)."""
        if self._model_forward is not None:
            return self._evaluate(self._model_forward, t)

        # One-sided at the shortest and longest quote, where some curves go flat.
        def numerical(x, h=1e-4):
            lo = np.clip(x - h, self.maturities[0], None)
            hi = np.clip(x + h, None, self.maturities[-1])
            return (self._zero_fn(hi) * hi - self._zero_fn(lo) * lo) / (hi - lo)

        return self._evaluate(numerical, t)

    def par(self, t):
        """Treasury par yield (%) implied by the curve: bill yield up to 6 months, semiannual coupon after."""
        def par_one(tenor):
            if tenor <= 0.5:
                return 2 * (self.discount(tenor) ** (-1 / (2 * tenor)) - 1) * 100
            coupon_times = np.arange(1, int(round(tenor * 2)) + 1) / 2
            discount = self.discount(coupon_times)
            return 2 * (1 - discount[-1]) / discount.sum() * 100

        return self._evaluate(lambda x: np.array([par_one(tenor) for tenor in x]), t)

    def curves(self, n_points=400):
        """Zero, forward and discount curves on an even grid between the shortest and longest quote."""
        grid = np.linspace(self.maturities[0], self.maturities[-1], n_points)
        return pd.DataFrame(
            {
                "zero_cc_pct": self.zero(grid),
                "forward_pct": self.forward(grid),
                "discount_factor": self.discount(grid),
            },
            index=pd.Index(grid, name="maturity_years"),
        )

    def is_arbitrage_free(self):
        """True if the forward rate stays positive, so the discount curve keeps falling."""
        return bool(self.curves()["forward_pct"].min() > 0)

    def figure(self):
        """Zero, forward and discount curves. Returns the Matplotlib figure."""
        curves = self.curves()
        fig, (ax_zero, ax_fwd, ax_df) = plt.subplots(1, 3, figsize=(16, 4.5))
        ax_zero.plot(curves.index, curves["zero_cc_pct"], label=self.name)
        ax_zero.plot(self.maturities, self.zeros, marker="o", linestyle="none", label="Bootstrapped zero")
        ax_zero.set_title("Zero curve")
        ax_zero.set_ylabel("Continuously compounded rate (%)")
        ax_fwd.plot(curves.index, curves["forward_pct"], color="C2")
        ax_fwd.axhline(0, color="grey", linewidth=0.8)
        ax_fwd.set_title("Instantaneous forward curve")
        ax_fwd.set_ylabel("Forward rate (%)")
        ax_df.plot(
            np.concatenate([[0.0], curves.index]),
            np.concatenate([[1.0], curves["discount_factor"]]),
            label=self.name,
        )
        ax_df.plot(self.maturities, self.quotes["discount_factor"], marker="o", linestyle="none", label="Bootstrapped zero")
        ax_df.set_title("Discount curve")
        ax_df.set_ylabel("Discount factor")
        for ax in (ax_zero, ax_fwd, ax_df):
            ax.set_xlabel("Maturity (years)")
            ax.grid(True, alpha=0.3)
        ax_zero.legend()
        ax_df.legend()
        as_of = self.as_of.strftime("%Y-%m-%d") if hasattr(self.as_of, "strftime") else self.as_of
        fig.suptitle(f"US Treasury Curves, {self.name} ({as_of})")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return fig

    def plot(self):
        """Zero curve with the bootstrapped quotes, forward curve and discount curve side by side."""
        self.figure()
        plt.show()

    def __repr__(self):
        as_of = self.as_of.strftime("%Y-%m-%d") if hasattr(self.as_of, "strftime") else self.as_of
        return f"YieldCurve(method={self.method!r}, as_of={as_of}, quotes={list(self.quotes.index)})"
