import numpy as np
from scipy import stats


def summarise_poisson_gamma(a, b, y, print_results=False):
    # 結果を保存する辞書
    results_dict = {}

    # 事後分布のパラメータ
    a_posterior = a + np.sum(y)
    b_posterior = b + len(y)
    results_dict["a_posterior"] = a_posterior
    results_dict["b_posterior"] = b_posterior

    # 事後平均
    posterior_mean = a_posterior / b_posterior
    results_dict["posterior_mean"] = posterior_mean

    # 事後分散
    posterior_var = a_posterior / (b_posterior**2)
    results_dict["posterior_var"] = posterior_var

    # 信用区間
    credible_interval = stats.gamma.interval(
        confidence=0.95, a=a_posterior, scale=1 / b_posterior
    )
    results_dict["credible_interval"] = credible_interval

    if print_results:
        print(
            "Posterior Distributiion:\n",
            f"\tgamma({a_posterior}, {b_posterior})",
            end="\n",
        )
        print("Posterior Mean:\n", f"\t {posterior_mean}", end="\n")
        print("Posterior Variance:\n", f"\t {posterior_var}", end="\n")
        print("95% Credible Interval:\n", f"\t {credible_interval}", end="\n")

    return results_dict


def summarise_binom_beta(a, b, y, n, print_results=False):
    # 結果を保存する辞書
    results_dict = {}

    # 事後分布のパラメータ
    a_posterior = a + y
    b_posterior = b + n - y
    results_dict["a_posterior"] = a_posterior
    results_dict["b_posterior"] = b_posterior

    # 事後平均, 事後分散
    posterior_mean, posterior_var = stats.beta.stats(
        a=a_posterior, b=b_posterior, moments="mv"
    )
    results_dict["posterior_mean"] = posterior_mean
    results_dict["posterior_var"] = posterior_var
    results_dict["posterior_std"] = np.sqrt(posterior_var)

    # 事後モード
    posterior_mode = (a_posterior - 1) / (a_posterior + b_posterior - 2)
    results_dict["posterior_mode"] = posterior_mode

    # 信用区間
    credible_interval = stats.beta.interval(
        confidence=0.95, a=a_posterior, b=b_posterior
    )
    results_dict["credible_interval"] = credible_interval

    if print_results:
        print(
            "Posterior Distributiion:\n",
            f"\t beta({a_posterior}, {b_posterior})",
            end="\n",
        )
        print("Posterior Mean:\n", f"\t {posterior_mean}", end="\n")
        print("Posterior Variance:\n", f"\t {posterior_var}", end="\n")
        print("Posterior Mode:\n", f"\t {posterior_mode}", end="\n")
        print("95% Credible Interval:\n", f"\t {credible_interval}", end="\n")

    return results_dict
