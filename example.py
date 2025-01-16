import matplotlib.pyplot as plt

from mfglib.alg import OccupationMeasureInclusion, OnlineMirrorDescent
from mfglib.env import Environment
from mfglib.metrics import exploitability_score

### Interesting observations to debug: OSQP seems to converge with larger obj than init when log_eps is large (10 or above?) and/or alpha is large (like 1000, 10000, etc.)
### T, n, etc. parameters seem to not matter much

# Environment (old trials)
# env_instance = Environment.beach_bar(log_eps=1000, T=1, n=3)
# env_instance = Environment.beach_bar(log_eps=10) # log_eps=1 or less OSQP obj decreases when convergel log_eps=10 or above OSQP obj increases when converge
# env_instance = Environment.beach_bar(T=10, n=10)

# env_instance = Environment.beach_bar(n=2,bar_loc=1,log_eps=1000)

### Environments (20241023 latest trials)
### in many cases, can even work with alpha=1.0 default without tuning (while OMD tuned still beaten)
### Also note that \eta=0 is generally preferred (and I have kicked it out from optuna for now but we can make these configurable in general as Anran suggested to Matteo)
### Note that it even works for mean-field dependent P and non-monotone cases it seems!!! like linear quadratic and random linear
### also seems in general big alpha works pretty well in many cases (just like OMD) -- but this is stepsize so a bit interesting?
### btw seems that when problem is too large (especially due to S and A; T seems to be less concerning or maybe just T alone not that concerning as size increase not that fast) the per-iteration cost of MF-OMI becomes much higher than OMD (but in some smaller problems like SIS MF-OMI is even faster)
### confirmed that tol become larger for osqp will accelerate a lot but also may even lead to infeasible policies and negative expl?
### in general shall we add an extra projection onto simplex of the resulting policy to be safe (so that expl evaluation makes sense)?
# env_instance = Environment.beach_bar() # work as is (after tuning?)
# env_instance = Environment.rock_paper_scissors() # fail
# env_instance = Environment.equilibrium_price() # work as is (after tuning?)
# env_instance = Environment.linear_quadratic() # work as is (after tuning?)
# env_instance = Environment.random_linear() # work as is (after tuning?) (and OMD failed miserably even with tuning)
# env_instance = Environment.susceptible_infected() # alpha=10 for MF-OMI works perfectly; alpha=1 is also fine just converging slowly; after tuning even better; OMD failed miserably even with tuning (btw default T is 50)
# env_instance = Environment.crowd_motion() # this one MF-OMI needs alpha=0.0001 or so to converge and OMD is faster
# env_instance = Environment.left_right() # both MF-OMI and OMD works well as is (after tuning)
# env_instance = Environment.conservative_treasure_hunting() # after tuning both MF-OMI and OMD can converge in 1-2 steps
env_instance = (
    Environment.building_evacuation()
)  # with alpha=0.0003 or 0.0001 MF-OMI converges slowly but well; OMD (even with tuning) start to blow up after approaching 1e-3 somehow

atol = None

rtol = 1e-8
max_iter = 1000
timeout = 300
n_trials = 30

### the following two can save some time in tuning/running (as 1e-8 is not achievable and 1000 iters are slow)
### especially for building evacuation and crowd motion (where MF-OMI converges but relatively slowly)
### note that when examples are relatively simple (so the above 1e-8+1000 tuning works)
###    tuning with below can lead to potentially controversial results compared to tuning with above
###    so a bit tricky what is a really fair comparison?
# rtol = 1e-2
# max_iter = 100
# timeout = 300
# n_trials = 10

rtol = 1e-3
max_iter = 1000
timeout = 300
n_trials = 10


omi = OccupationMeasureInclusion(
    alpha=0.0003
)  # alpha=10 # alpha=0.0004 # alpha=0.0001 # alpha=0.0003
# omi_tuned = omi.tune([env_instance], atol=atol, rtol=rtol, max_iter=max_iter, timeout=timeout, n_trials=n_trials,)
omi_tuned = omi

solns, expls, runtimes = omi_tuned.solve(
    env_instance, max_iter=max_iter, verbose=True, atol=atol, rtol=rtol
)

# solns, expls, runtimes = OccupationMeasureInclusion(alpha=1e-4, eta=1e-4).solve(env_instance, max_iter=1000, verbose=True, atol=1e-8, rtol=1e-8)
# solns, expls, runtimes = OnlineMirrorDescent(alpha=0.01).solve(env_instance, max_iter=1000, verbose=True, atol=1e-5, rtol=1e-5)

# TODO: Add optuna test

### OMD
omd = OnlineMirrorDescent()
omd_tuned = omd.tune(
    [env_instance],
    atol=atol,
    rtol=rtol,
    max_iter=max_iter,
    timeout=timeout,
    n_trials=n_trials,
)
# omd_tuned = omd

solns_omd, expls_omd, runtimes_omd = omd_tuned.solve(
    env_instance, max_iter=max_iter, verbose=True, atol=atol, rtol=rtol
)

# plt.semilogy(runtimes, exploitability_score(env_instance, solns), label="mfomi")
# plt.semilogy(runtimes_omd, exploitability_score(env_instance, solns_omd), label="omd")
# plt.grid(True)
# plt.xlabel("Runtime (seconds)")
# plt.ylabel("Exploitability")
# # plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
# plt.show()

plt.semilogy(exploitability_score(env_instance, solns), label="mfomi")
plt.semilogy(exploitability_score(env_instance, solns_omd), label="omd")
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Exploitability")
# plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.legend()
plt.show()
