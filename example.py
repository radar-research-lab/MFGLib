from mfglib.env import Environment
from mfglib.alg import OccupationMeasureInclusion, OnlineMirrorDescent
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt

### Interesting observations to debug: OSQP seems to converge with larger obj than init when log_eps is large (10 or above?) and/or alpha is large (like 1000, 10000, etc.)
### T, n, etc. parameters seem to not matter much

# Environment
# env_instance = Environment.beach_bar(log_eps=1000, T=1, n=3)
# env_instance = Environment.beach_bar(log_eps=10) # log_eps=1 or less OSQP obj decreases when convergel log_eps=10 or above OSQP obj increases when converge
# env_instance = Environment.beach_bar(T=10, n=10)

env_instance = Environment.beach_bar(n=2,bar_loc=1,log_eps=1000)

# env_instance = Environment.rock_paper_scissors()
# env_instance = Environment.equilibrium_price()
# env_instance = Environment.linear_quadratic()
# env_instance = Environment.random_linear()
# env_instance = Environment.susceptible_infected()
# env_instance = Environment.crowd_motion()
# env_instance = Environment.left_right()
# env_instance = Environment.conservative_treasure_hunting
# env_instance = Environment.building_evacuation()

atol = None
rtol = 1e-1
max_iter = 1000
timeout = 300
n_trials = 100

omi = OccupationMeasureInclusion()
omi_tuned = omi.tune([env_instance], atol=atol, rtol=rtol, max_iter=max_iter, timeout=timeout, n_trials=n_trials,)

solns, expls, runtimes = omi_tuned.solve(env_instance, max_iter=max_iter, verbose=True, atol=atol, rtol=rtol)

# solns, expls, runtimes = OccupationMeasureInclusion(alpha=1e-4, eta=1e-4).solve(env_instance, max_iter=1000, verbose=True, atol=1e-8, rtol=1e-8)
# solns, expls, runtimes = OnlineMirrorDescent(alpha=0.01).solve(env_instance, max_iter=1000, verbose=True, atol=1e-5, rtol=1e-5)

# TODO: Add optuna test

plt.semilogy(runtimes, exploitability_score(env_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
# plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()
