from mfglib.env import Environment
from mfglib.alg import OccupationMeasureInclusion, OnlineMirrorDescent
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt

### Interesting observations to debug: OSQP seems to converge with larger obj than init when log_eps is large (10 or above?) and/or alpha is large (like 1000, 10000, etc.)
### T, n, etc. parameters seem to not matter much

# Environment
# env_instance = Environment.beach_bar(log_eps=1000, T=1, n=3)
env_instance = Environment.beach_bar(log_eps=10) # log_eps=1 or less OSQP obj decreases when convergel log_eps=10 or above OSQP obj increases when converge

solns, expls, runtimes = OccupationMeasureInclusion(alpha=5e-2, eta=0).solve(env_instance, max_iter=10000, verbose=True, atol=1e-8, rtol=1e-8)
# solns, expls, runtimes = OnlineMirrorDescent(alpha=0.01).solve(env_instance, max_iter=1000, verbose=True, atol=1e-5, rtol=1e-5)

# TODO: Add optuna test

plt.semilogy(runtimes, exploitability_score(env_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
# plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()
