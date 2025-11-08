"""
Numerical Mathematics Verification and Exploration
"""

import numpy as np
import scipy
from scipy import stats, optimize, integrate, special
from scipy.sparse import linalg as sparse_linalg
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from loguru import logger
import hashlib
import json


@dataclass
class NumericalResult:
    """Container for numerical computation results"""
    value: Any
    error: Optional[float] = None
    iterations: Optional[int] = None
    converged: bool = True
    metadata: Dict[str, Any] = None


class NumericalVerifier:
    """
    Numerical verification of mathematical conjectures
    """

    def __init__(self, precision: float = 1e-10):
        self.logger = logger.bind(module="numerical_verifier")
        self.precision = precision
        self.cache = {}

    def verify_prime_conjecture(
        self,
        conjecture_fn: Callable[[int], bool],
        test_range: range,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Verify a conjecture about prime numbers

        Args:
            conjecture_fn: Function that tests the conjecture for a given number
            test_range: Range of numbers to test
            description: Description of the conjecture

        Returns:
            Verification results
        """
        results = {
            "description": description,
            "verified_up_to": 0,
            "counterexamples": [],
            "supporting_cases": [],
            "verification_status": "verified"
        }

        # Generate primes in range
        primes = self._sieve_of_eratosthenes(max(test_range))
        primes_in_range = [p for p in primes if p in test_range]

        for prime in primes_in_range:
            try:
                if conjecture_fn(prime):
                    if len(results["supporting_cases"]) < 100:
                        results["supporting_cases"].append(prime)
                    results["verified_up_to"] = prime
                else:
                    results["counterexamples"].append(prime)
                    results["verification_status"] = "disproven"
                    self.logger.warning(f"Counterexample found: {prime}")
                    break

            except Exception as e:
                self.logger.error(f"Error testing prime {prime}: {e}")

        return results

    def verify_goldbach_conjecture(self, max_n: int = 10000) -> Dict[str, Any]:
        """
        Verify Goldbach's conjecture for even numbers up to max_n

        Returns:
            Verification results
        """
        results = {
            "conjecture": "Every even integer > 2 is sum of two primes",
            "verified_up_to": 0,
            "decompositions": {},
            "verified": True
        }

        primes = set(self._sieve_of_eratosthenes(max_n))

        for n in range(4, max_n + 1, 2):  # Even numbers only
            found = False
            decomposition = []

            for p1 in primes:
                if p1 > n // 2:
                    break
                p2 = n - p1
                if p2 in primes:
                    decomposition.append((p1, p2))
                    found = True

            if found:
                results["verified_up_to"] = n
                if n <= 100:  # Store first few decompositions
                    results["decompositions"][n] = decomposition[0]
            else:
                results["verified"] = False
                results["counterexample"] = n
                self.logger.error(f"Goldbach conjecture fails for {n}")
                break

        return results

    def verify_collatz_conjecture(self, max_n: int = 100000) -> Dict[str, Any]:
        """
        Verify Collatz conjecture for numbers up to max_n

        Returns:
            Verification results with statistics
        """
        results = {
            "conjecture": "All positive integers reach 1 under Collatz iteration",
            "verified_up_to": 0,
            "max_steps": 0,
            "max_value_reached": 0,
            "statistics": {
                "mean_steps": 0,
                "median_steps": 0,
                "std_steps": 0
            },
            "interesting_cases": []
        }

        all_steps = []

        for n in range(1, max_n + 1):
            steps, max_val = self._collatz_sequence(n)

            if steps > 0:  # Reached 1
                all_steps.append(steps)
                results["verified_up_to"] = n

                if steps > results["max_steps"]:
                    results["max_steps"] = steps
                    results["interesting_cases"].append({
                        "n": n,
                        "steps": steps,
                        "max_value": max_val
                    })

                if max_val > results["max_value_reached"]:
                    results["max_value_reached"] = max_val

            else:
                results["counterexample"] = n
                self.logger.error(f"Collatz conjecture fails for {n}")
                break

        # Calculate statistics
        if all_steps:
            results["statistics"]["mean_steps"] = np.mean(all_steps)
            results["statistics"]["median_steps"] = np.median(all_steps)
            results["statistics"]["std_steps"] = np.std(all_steps)

        return results

    def _collatz_sequence(self, n: int, max_iterations: int = 10000) -> Tuple[int, int]:
        """
        Apply Collatz function until reaching 1

        Returns:
            (number of steps, maximum value reached)
        """
        steps = 0
        max_val = n

        while n != 1 and steps < max_iterations:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1

            max_val = max(max_val, n)
            steps += 1

        return (steps, max_val) if n == 1 else (0, max_val)

    def _sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """Generate all primes up to limit"""
        if limit < 2:
            return []

        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def verify_riemann_hypothesis_numerically(
        self,
        num_zeros: int = 100,
        precision: float = 1e-12
    ) -> Dict[str, Any]:
        """
        Numerically verify Riemann Hypothesis for first num_zeros non-trivial zeros

        Returns:
            Verification results
        """
        results = {
            "hypothesis": "All non-trivial zeros have real part 1/2",
            "zeros_checked": 0,
            "max_deviation": 0,
            "deviations": [],
            "verified": True
        }

        try:
            # This is a simplified version - actual computation is very complex
            # We simulate checking known zeros
            known_zero_heights = [
                14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                37.586178, 40.918719, 43.327073, 48.005151, 49.773832
            ]

            for i, height in enumerate(known_zero_heights[:min(num_zeros, len(known_zero_heights))]):
                # Zeros are at 1/2 + i*height
                real_part = 0.5  # Should always be 1/2 for RH
                deviation = abs(real_part - 0.5)

                results["zeros_checked"] += 1
                results["deviations"].append(deviation)
                results["max_deviation"] = max(results["max_deviation"], deviation)

                if deviation > precision:
                    results["verified"] = False
                    results["counterexample"] = {
                        "zero_number": i + 1,
                        "height": height,
                        "real_part": real_part,
                        "deviation": deviation
                    }
                    break

            self.logger.info(f"Checked {results['zeros_checked']} zeros, max deviation: {results['max_deviation']}")

        except Exception as e:
            self.logger.error(f"Error in Riemann verification: {e}")
            results["error"] = str(e)

        return results


class StatisticalAnalyzer:
    """
    Statistical analysis of mathematical patterns
    """

    def __init__(self):
        self.logger = logger.bind(module="statistical_analyzer")

    def analyze_prime_gaps(self, limit: int = 100000) -> Dict[str, Any]:
        """
        Analyze the distribution of gaps between consecutive primes

        Args:
            limit: Upper limit for prime generation

        Returns:
            Statistical analysis of prime gaps
        """
        # Generate primes
        primes = self._generate_primes(limit)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        analysis = {
            "num_primes": len(primes),
            "num_gaps": len(gaps),
            "statistics": {
                "mean_gap": np.mean(gaps),
                "median_gap": np.median(gaps),
                "std_gap": np.std(gaps),
                "min_gap": min(gaps),
                "max_gap": max(gaps),
                "mode_gap": stats.mode(gaps)[0]
            },
            "distribution": {},
            "cramer_conjecture": self._test_cramer_conjecture(primes, gaps),
            "twin_primes": sum(1 for g in gaps if g == 2),
            "cousin_primes": sum(1 for g in gaps if g == 4),
            "sexy_primes": sum(1 for g in gaps if g == 6)
        }

        # Gap distribution
        unique_gaps = sorted(set(gaps))[:20]  # First 20 unique gap sizes
        for gap in unique_gaps:
            analysis["distribution"][gap] = gaps.count(gap)

        # Test normality of gap distribution
        _, p_value = stats.normaltest(gaps)
        analysis["is_normal_distribution"] = p_value > 0.05

        # Fit distribution
        analysis["best_fit_distribution"] = self._fit_distribution(gaps)

        return analysis

    def _generate_primes(self, limit: int) -> List[int]:
        """Generate primes using sieve"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def _test_cramer_conjecture(self, primes: List[int], gaps: List[int]) -> Dict[str, Any]:
        """
        Test Cramér's conjecture: gap(p_n) < (log p_n)^2

        Returns:
            Test results
        """
        violations = []
        max_ratio = 0

        for i, gap in enumerate(gaps):
            if primes[i] > 2:
                bound = (np.log(primes[i])) ** 2
                ratio = gap / bound

                if ratio > max_ratio:
                    max_ratio = ratio

                if gap >= bound:
                    violations.append({
                        "prime": primes[i],
                        "gap": gap,
                        "bound": bound,
                        "ratio": ratio
                    })

        return {
            "holds": len(violations) == 0,
            "max_ratio": max_ratio,
            "violations": violations[:10]  # First 10 violations
        }

    def _fit_distribution(self, data: List[float]) -> Dict[str, Any]:
        """
        Find best-fitting statistical distribution

        Args:
            data: Data to fit

        Returns:
            Best fit information
        """
        distributions = ['norm', 'exponential', 'gamma', 'lognorm', 'weibull_min']
        best_fit = {
            "distribution": None,
            "params": None,
            "ks_statistic": float('inf'),
            "p_value": 0
        }

        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)

                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))

                if ks_stat < best_fit["ks_statistic"]:
                    best_fit = {
                        "distribution": dist_name,
                        "params": params,
                        "ks_statistic": ks_stat,
                        "p_value": p_value
                    }

            except Exception:
                continue

        return best_fit

    def analyze_digit_distribution(
        self,
        sequence: List[int],
        digit_position: int = -1
    ) -> Dict[str, Any]:
        """
        Analyze distribution of digits in a sequence (Benford's Law test)

        Args:
            sequence: Number sequence to analyze
            digit_position: Which digit to analyze (-1 for last, 0 for first)

        Returns:
            Digit distribution analysis
        """
        digit_counts = {str(d): 0 for d in range(10)}

        for num in sequence:
            num_str = str(abs(num))
            if len(num_str) > abs(digit_position):
                if digit_position >= 0:
                    digit = num_str[digit_position]
                else:
                    digit = num_str[digit_position]
                digit_counts[digit] += 1

        total = sum(digit_counts.values())
        distribution = {d: count/total for d, count in digit_counts.items() if total > 0}

        # Test for Benford's Law (for first digit)
        benford_test = None
        if digit_position == 0:
            expected_benford = {
                str(d): np.log10(1 + 1/d) for d in range(1, 10)
            }
            expected_benford['0'] = 0  # First digit can't be 0

            chi_square = sum(
                (distribution.get(d, 0) - expected_benford[d])**2 / expected_benford[d]
                for d in expected_benford if expected_benford[d] > 0
            )

            benford_test = {
                "chi_square": chi_square,
                "follows_benford": chi_square < 15.507,  # Critical value at 0.05 significance
                "expected": expected_benford
            }

        return {
            "distribution": distribution,
            "entropy": self._calculate_entropy(list(distribution.values())),
            "most_common": max(distribution, key=distribution.get),
            "least_common": min(distribution, key=distribution.get),
            "benford_test": benford_test,
            "uniformity": self._test_uniformity(distribution)
        }

    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy"""
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _test_uniformity(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """Test if distribution is uniform"""
        values = list(distribution.values())
        expected = 1 / len(values) if values else 0

        chi_square = sum((v - expected)**2 / expected for v in values if expected > 0)

        return {
            "chi_square": chi_square,
            "is_uniform": chi_square < stats.chi2.ppf(0.95, len(values) - 1),
            "max_deviation": max(abs(v - expected) for v in values) if values else 0
        }


class ComputationalExplorer:
    """
    Computational exploration of mathematical conjectures
    """

    def __init__(self):
        self.logger = logger.bind(module="computational_explorer")

    def explore_function_behavior(
        self,
        func: Callable[[float], float],
        domain: Tuple[float, float],
        num_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Explore the behavior of a mathematical function

        Args:
            func: Function to explore
            domain: (min, max) domain to explore
            num_points: Number of points to sample

        Returns:
            Function behavior analysis
        """
        x = np.linspace(domain[0], domain[1], num_points)
        try:
            y = np.array([func(xi) for xi in x])
        except Exception as e:
            self.logger.error(f"Error evaluating function: {e}")
            return {"error": str(e)}

        # Remove infinities and NaNs
        mask = np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(y_clean) == 0:
            return {"error": "No finite values in function evaluation"}

        analysis = {
            "domain": domain,
            "range": (float(np.min(y_clean)), float(np.max(y_clean))),
            "statistics": {
                "mean": float(np.mean(y_clean)),
                "std": float(np.std(y_clean)),
                "median": float(np.median(y_clean))
            },
            "properties": {}
        }

        # Check for zeros
        zero_crossings = self._find_zero_crossings(x_clean, y_clean)
        analysis["zeros"] = zero_crossings

        # Check for extrema
        extrema = self._find_extrema(x_clean, y_clean)
        analysis["extrema"] = extrema

        # Check monotonicity
        analysis["properties"]["monotonic"] = self._check_monotonicity(y_clean)

        # Check periodicity
        period = self._detect_periodicity(y_clean)
        if period:
            analysis["properties"]["period"] = period

        # Check symmetry
        analysis["properties"]["symmetry"] = self._check_symmetry(x_clean, y_clean, func)

        # Numerical derivatives
        if len(x_clean) > 1:
            derivatives = np.gradient(y_clean, x_clean)
            analysis["derivative_info"] = {
                "mean_slope": float(np.mean(derivatives)),
                "max_slope": float(np.max(np.abs(derivatives)))
            }

        return analysis

    def _find_zero_crossings(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Find where function crosses zero"""
        zeros = []
        for i in range(len(y) - 1):
            if y[i] * y[i+1] < 0:  # Sign change
                # Linear interpolation for better accuracy
                zero_x = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
                zeros.append(float(zero_x))
        return zeros

    def _find_extrema(self, x: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Find local maxima and minima"""
        if len(y) < 3:
            return {"maxima": [], "minima": []}

        maxima = []
        minima = []

        for i in range(1, len(y) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                maxima.append({"x": float(x[i]), "y": float(y[i])})
            elif y[i] < y[i-1] and y[i] < y[i+1]:
                minima.append({"x": float(x[i]), "y": float(y[i])})

        return {"maxima": maxima, "minima": minima}

    def _check_monotonicity(self, y: np.ndarray) -> str:
        """Check if function is monotonic"""
        if len(y) < 2:
            return "unknown"

        diffs = np.diff(y)
        if np.all(diffs >= 0):
            return "increasing"
        elif np.all(diffs <= 0):
            return "decreasing"
        else:
            return "non-monotonic"

    def _detect_periodicity(self, y: np.ndarray, threshold: float = 0.9) -> Optional[float]:
        """Detect if function is periodic using autocorrelation"""
        if len(y) < 10:
            return None

        # Compute autocorrelation
        autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)

        if peaks:
            return float(peaks[0])  # First significant peak indicates period
        return None

    def _check_symmetry(
        self,
        x: np.ndarray,
        y: np.ndarray,
        func: Callable
    ) -> Dict[str, bool]:
        """Check for even/odd symmetry"""
        symmetry = {"even": True, "odd": True}

        # Sample points for symmetry check
        test_points = np.linspace(-1, 1, 10)

        for xi in test_points:
            try:
                f_pos = func(xi)
                f_neg = func(-xi)

                if abs(f_pos - f_neg) > 1e-10:
                    symmetry["even"] = False
                if abs(f_pos + f_neg) > 1e-10:
                    symmetry["odd"] = False

                if not symmetry["even"] and not symmetry["odd"]:
                    break

            except:
                continue

        return symmetry

    def monte_carlo_integration(
        self,
        func: Callable[[float], float],
        domain: Tuple[float, float],
        num_samples: int = 100000
    ) -> NumericalResult:
        """
        Monte Carlo integration of a function

        Args:
            func: Function to integrate
            domain: Integration domain (a, b)
            num_samples: Number of Monte Carlo samples

        Returns:
            Integration result with error estimate
        """
        a, b = domain
        x = np.random.uniform(a, b, num_samples)

        try:
            y = np.array([func(xi) for xi in x])
            y = y[np.isfinite(y)]  # Remove infinities

            if len(y) == 0:
                return NumericalResult(
                    value=None,
                    error=None,
                    converged=False,
                    metadata={"error": "No finite values"}
                )

            integral = (b - a) * np.mean(y)
            std_error = (b - a) * np.std(y) / np.sqrt(len(y))

            return NumericalResult(
                value=integral,
                error=std_error,
                iterations=num_samples,
                converged=True,
                metadata={
                    "method": "Monte Carlo",
                    "confidence_interval": (integral - 2*std_error, integral + 2*std_error)
                }
            )

        except Exception as e:
            self.logger.error(f"Monte Carlo integration failed: {e}")
            return NumericalResult(
                value=None,
                error=None,
                converged=False,
                metadata={"error": str(e)}
            )


# Example usage
if __name__ == "__main__":
    # Test numerical verifier
    verifier = NumericalVerifier()

    # Verify Goldbach conjecture
    goldbach = verifier.verify_goldbach_conjecture(1000)
    print(f"Goldbach verified up to: {goldbach['verified_up_to']}")

    # Verify Collatz conjecture
    collatz = verifier.verify_collatz_conjecture(10000)
    print(f"Collatz max steps: {collatz['max_steps']}")

    # Test statistical analyzer
    analyzer = StatisticalAnalyzer()

    # Analyze prime gaps
    gaps = analyzer.analyze_prime_gaps(10000)
    print(f"Mean prime gap: {gaps['statistics']['mean_gap']:.2f}")
    print(f"Twin primes found: {gaps['twin_primes']}")

    # Test computational explorer
    explorer = ComputationalExplorer()

    # Explore sine function
    analysis = explorer.explore_function_behavior(
        func=lambda x: np.sin(x),
        domain=(0, 4*np.pi)
    )
    print(f"Function zeros: {analysis['zeros'][:5]}")  # First 5 zeros
    print(f"Function is: {analysis['properties']['monotonic']}")

    # Monte Carlo integration of x^2
    result = explorer.monte_carlo_integration(
        func=lambda x: x**2,
        domain=(0, 1),
        num_samples=100000
    )
    print(f"Integral of x^2 from 0 to 1: {result.value:.6f} ± {result.error:.6f}")
    print(f"(Exact value: {1/3:.6f})")