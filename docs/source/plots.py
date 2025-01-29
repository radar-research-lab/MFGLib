import matplotlib.pyplot as plt

from mfglib.alg import MFOMO, FictitiousPlay, OnlineMirrorDescent, PriorDescent
from mfglib.env import Environment


def plot_beach_bar_exploitability() -> None:
    beach_bar = Environment.beach_bar()
    online_mirror_descent = OnlineMirrorDescent()

    _, expls, _ = online_mirror_descent.solve(beach_bar)

    plt.figure(figsize=(8, 8))
    plt.semilogy(expls, label="Online Mirror Descent")
    plt.legend(loc=0)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Beach Bar Environment")
    plt.show()


def plot_fictitious_play() -> None:
    plt.figure(figsize=(8, 8))

    rock_paper_scissors = Environment.rock_paper_scissors()
    for alpha in [0.1, 0.5, 0.75, None]:
        _, expls, _ = FictitiousPlay(alpha=alpha).solve(
            env_instance=rock_paper_scissors,
            max_iter=300,
            atol=None,
            rtol=None,
        )
        plt.semilogy(expls, label=f"alpha: {alpha}")

    plt.legend(loc=3)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Rock Paper Scissors Environment - Fictitious Play Algorithm")
    plt.show()


def plot_online_mirror_descent() -> None:
    plt.figure(figsize=(8, 8))

    rock_paper_scissors = Environment.rock_paper_scissors()
    for alpha in [0.01, 0.1, 1.0, 10]:
        _, expls, _ = OnlineMirrorDescent(alpha=alpha).solve(
            env_instance=rock_paper_scissors,
            max_iter=300,
            atol=None,
            rtol=None,
        )
        plt.semilogy(expls, label=f"alpha: {alpha}")

    plt.legend(loc=3)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Rock Paper Scissors Environment - Online Mirror Descent Algorithm")
    plt.show()


def plot_prior_descent() -> None:
    plt.figure(figsize=(8, 8))

    rock_paper_scissors = Environment.rock_paper_scissors()

    eta_values = [0.01, 0.1, 1.0, 10]
    n_inner_values = [None, 100, 20, 5]

    for eta, n_inner in zip(eta_values, n_inner_values):
        _, expls, _ = PriorDescent(eta=eta, n_inner=n_inner).solve(
            env_instance=rock_paper_scissors,
            max_iter=300,
            atol=None,
            rtol=None,
        )
        plt.semilogy(expls, label=f"eta: {eta}, n_inner: {n_inner}")

    plt.legend(loc=3)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Rock Paper Scissors Environment - Prior Descent Algorithm")
    plt.show()


def plot_mf_omo() -> None:
    plt.figure(figsize=(8, 8))

    rock_paper_scissors = Environment.rock_paper_scissors()

    lrs = [0.01, 0.1, 1, 10]

    for lr in lrs:
        opt = {"name": "Adam", "config": {"lr": lr}}
        _, expls, _ = MFOMO(optimizer=opt).solve(
            env_instance=rock_paper_scissors,
            max_iter=300,
            atol=None,
            rtol=None,
        )
        plt.semilogy(expls, label=f"Adam optimizer - lr: {lr}")

    plt.legend(loc=3)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
    plt.show()
