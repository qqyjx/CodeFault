"""
Statistical Testing Utilities for CodeFault Experiments

Provides functions for:
- Wilcoxon signed-rank test (paired samples)
- Mann-Whitney U test (independent samples)
- Effect size calculations (Cliff's Delta, Cohen's d)
- Confidence interval estimation
- Multiple comparison correction (Bonferroni, Holm)

References:
- SciPy Statistics: https://docs.scipy.org/doc/scipy/reference/stats.html
- Cliff's Delta: https://en.wikipedia.org/wiki/Effect_size#Cliff's_delta
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def wilcoxon_signed_rank_test(
    scores1: List[float],
    scores2: List[float],
    alternative: str = 'greater'
) -> Dict:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Use this when comparing CodeFault vs. baseline on the same test set
    across multiple runs.

    Args:
        scores1: Scores from method 1 (e.g., CodeFault AUCs from 5 runs)
        scores2: Scores from method 2 (e.g., DeepJIT AUCs from 5 runs)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Dict with statistic, p_value, and significance markers
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have the same length")

    if len(scores1) < 5:
        # For very small samples, use exact method
        method = 'exact'
    else:
        method = 'auto'

    try:
        statistic, p_value = stats.wilcoxon(
            scores1, scores2,
            alternative=alternative,
            method=method
        )
    except ValueError as e:
        # Handle case where all differences are zero
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant_0.05': False,
            'significant_0.01': False,
            'significant_0.001': False,
            'significance_marker': '',
            'error': str(e)
        }

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001,
        'significance_marker': get_significance_marker(p_value)
    }


def mann_whitney_u_test(
    scores1: List[float],
    scores2: List[float],
    alternative: str = 'greater'
) -> Dict:
    """
    Perform Mann-Whitney U test for independent samples.

    Use this when comparing two independent groups.

    Args:
        scores1: Scores from group 1
        scores2: Scores from group 2
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Dict with statistic, p_value, and significance markers
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    statistic, p_value = stats.mannwhitneyu(
        scores1, scores2,
        alternative=alternative
    )

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001,
        'significance_marker': get_significance_marker(p_value)
    }


def cliffs_delta(scores1: List[float], scores2: List[float]) -> Dict:
    """
    Calculate Cliff's Delta effect size.

    Cliff's Delta is a non-parametric effect size measure that quantifies
    the amount of difference between two groups.

    Interpretation:
        |d| < 0.147: negligible
        |d| < 0.33: small
        |d| < 0.474: medium
        |d| >= 0.474: large

    Args:
        scores1: Scores from group 1
        scores2: Scores from group 2

    Returns:
        Dict with delta value and interpretation
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    n1, n2 = len(scores1), len(scores2)

    # Count dominance
    more = 0
    less = 0
    for x1 in scores1:
        for x2 in scores2:
            if x1 > x2:
                more += 1
            elif x1 < x2:
                less += 1

    delta = (more - less) / (n1 * n2)

    # Interpret effect size
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        'delta': float(delta),
        'abs_delta': float(abs_delta),
        'interpretation': interpretation
    }


def cohens_d(scores1: List[float], scores2: List[float]) -> Dict:
    """
    Calculate Cohen's d effect size.

    Interpretation:
        |d| < 0.2: small
        |d| < 0.5: medium
        |d| < 0.8: large
        |d| >= 0.8: very large

    Args:
        scores1: Scores from group 1
        scores2: Scores from group 2

    Returns:
        Dict with d value and interpretation
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    n1, n2 = len(scores1), len(scores2)
    var1, var2 = scores1.var(), scores2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (scores1.mean() - scores2.mean()) / pooled_std

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "small"
    elif abs_d < 0.5:
        interpretation = "medium"
    elif abs_d < 0.8:
        interpretation = "large"
    else:
        interpretation = "very large"

    return {
        'd': float(d),
        'abs_d': float(abs_d),
        'interpretation': interpretation
    }


def confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Dict:
    """
    Calculate confidence interval for mean.

    Args:
        scores: Sample scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dict with mean, std, ci_lower, ci_upper
    """
    scores = np.array(scores)
    n = len(scores)
    mean = scores.mean()
    std = scores.std(ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)    # Standard error of mean

    # t-value for given confidence level
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * sem

    return {
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'ci_lower': float(mean - margin),
        'ci_upper': float(mean + margin),
        'confidence': confidence
    }


def get_significance_marker(p_value: float) -> str:
    """
    Get significance marker for p-value.

    Returns:
        '***' for p < 0.001
        '**' for p < 0.01
        '*' for p < 0.05
        '' otherwise
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level

    Returns:
        Dict with corrected alpha and significant flags
    """
    n = len(p_values)
    corrected_alpha = alpha / n

    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_comparisons': n,
        'significant': [p < corrected_alpha for p in p_values],
        'p_values': p_values
    }


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Less conservative than Bonferroni.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level

    Returns:
        Dict with sorted p-values and significant flags
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    significant = [False] * n
    for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvals)):
        adjusted_alpha = alpha / (n - i)
        if pval < adjusted_alpha:
            significant[idx] = True
        else:
            break  # Stop once we fail to reject

    return {
        'original_alpha': alpha,
        'n_comparisons': n,
        'significant': significant,
        'p_values': p_values
    }


def compare_models(
    codefault_results: List[Dict],
    baseline_results: List[Dict],
    metric: str = 'auc_roc',
    baseline_name: str = 'Baseline'
) -> Dict:
    """
    Complete statistical comparison between CodeFault and a baseline.

    Args:
        codefault_results: List of result dicts from CodeFault runs
        baseline_results: List of result dicts from baseline runs
        metric: Metric to compare (e.g., 'auc_roc', 'f1')
        baseline_name: Name of baseline for reporting

    Returns:
        Comprehensive comparison results
    """
    cf_scores = [r[metric] for r in codefault_results]
    bl_scores = [r[metric] for r in baseline_results]

    # Statistics
    cf_ci = confidence_interval(cf_scores)
    bl_ci = confidence_interval(bl_scores)

    # Tests
    wilcoxon_result = wilcoxon_signed_rank_test(cf_scores, bl_scores)
    effect_size = cliffs_delta(cf_scores, bl_scores)

    # Improvement
    improvement = (cf_ci['mean'] - bl_ci['mean']) / bl_ci['mean'] * 100

    return {
        'metric': metric,
        'baseline': baseline_name,
        'codefault': {
            'mean': cf_ci['mean'],
            'std': cf_ci['std'],
            'ci_95': (cf_ci['ci_lower'], cf_ci['ci_upper'])
        },
        'baseline_stats': {
            'mean': bl_ci['mean'],
            'std': bl_ci['std'],
            'ci_95': (bl_ci['ci_lower'], bl_ci['ci_upper'])
        },
        'improvement_percent': improvement,
        'wilcoxon': wilcoxon_result,
        'effect_size': effect_size,
        'summary': f"{cf_ci['mean']:.3f}±{cf_ci['std']:.3f}{wilcoxon_result['significance_marker']}"
    }


def format_result_for_latex(
    mean: float,
    std: float,
    significance_marker: str = '',
    bold: bool = False
) -> str:
    """
    Format result for LaTeX table.

    Args:
        mean: Mean value
        std: Standard deviation
        significance_marker: '', '*', '**', or '***'
        bold: Whether to bold the result

    Returns:
        LaTeX formatted string like "0.876±0.012***" or "\\textbf{0.876±0.012}***"
    """
    formatted = f"{mean:.3f}±{std:.3f}"
    if bold:
        formatted = f"\\textbf{{{formatted}}}"
    if significance_marker:
        formatted += f"$^{{{significance_marker}}}$"
    return formatted


if __name__ == "__main__":
    # Example usage
    print("Statistical Testing Utilities Demo\n")

    # Simulated results from 5 runs
    codefault_aucs = [0.872, 0.879, 0.881, 0.874, 0.876]
    deepjit_aucs = [0.808, 0.815, 0.812, 0.810, 0.814]

    print("CodeFault AUCs:", codefault_aucs)
    print("DeepJIT AUCs:", deepjit_aucs)
    print()

    # Confidence intervals
    cf_ci = confidence_interval(codefault_aucs)
    dj_ci = confidence_interval(deepjit_aucs)
    print(f"CodeFault: {cf_ci['mean']:.3f} ± {cf_ci['std']:.3f} (95% CI: [{cf_ci['ci_lower']:.3f}, {cf_ci['ci_upper']:.3f}])")
    print(f"DeepJIT: {dj_ci['mean']:.3f} ± {dj_ci['std']:.3f} (95% CI: [{dj_ci['ci_lower']:.3f}, {dj_ci['ci_upper']:.3f}])")
    print()

    # Wilcoxon test
    wilcoxon = wilcoxon_signed_rank_test(codefault_aucs, deepjit_aucs)
    print(f"Wilcoxon test: statistic={wilcoxon['statistic']:.3f}, p={wilcoxon['p_value']:.4f} {wilcoxon['significance_marker']}")
    print()

    # Effect size
    effect = cliffs_delta(codefault_aucs, deepjit_aucs)
    print(f"Cliff's Delta: {effect['delta']:.3f} ({effect['interpretation']})")
    print()

    # LaTeX output
    print("LaTeX format:")
    print(f"  CodeFault: {format_result_for_latex(cf_ci['mean'], cf_ci['std'], wilcoxon['significance_marker'], bold=True)}")
    print(f"  DeepJIT: {format_result_for_latex(dj_ci['mean'], dj_ci['std'])}")
