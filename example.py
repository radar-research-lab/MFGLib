from mfglib.env import Environment
from mfglib.alg import OccupationMeasureInclusion
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt

# Environment
env_instance = Environment.beach_bar()

solns, expls, runtimes = OccupationMeasureInclusion(alpha=1e-3, eta=1e-2).solve(env_instance, max_iter=10000, verbose=True)

# TODO: Add optuna test

plt.semilogy(runtimes, exploitability_score(env_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()
