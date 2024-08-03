from mfglib.env import Environment
from mfglib.alg import OccupationMeasureInclusion
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt

# Environment
rock_paper_scissors_instance = Environment.rock_paper_scissors()

solns, expls, runtimes = OccupationMeasureInclusion(alpha=1e-3, eta=1e-5).solve(rock_paper_scissors_instance, max_iter=300, verbose=True)

# TODO: Add optuna test

plt.semilogy(runtimes, exploitability_score(rock_paper_scissors_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()
