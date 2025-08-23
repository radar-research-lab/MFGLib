import matplotlib.pyplot as plt
import optuna

from mfglib.alg import OccupationMeasureInclusion
from mfglib.env import Environment
from mfglib.tuning import GeometricMean

# By default, optuna displays logs. This silences them.
optuna.logging.set_verbosity(optuna.logging.WARNING)

env = Environment.rock_paper_scissors()

### Initialize algorithm arbitrarily

### this corresponds to 1e-8 osqp_atol and osqp_rtol? but eventual constraint violation seems to be even more based on the checking below?
### turns out to be mainly due to something like -1e-11, -1e-11, 2e-11 and then add up to 2e-14, and hence getting some -1500+ stuff entries. 
### but weirdly/luckily the exploitability looks small/not wild/big indeed for such invalid pi
### d is indeed okay; just need to set some arg to overwrite to 0 if d entries abs < thresh
### not really, this defaults to atol and rtol and they default to 1e-3.
# alg_orig = OccupationMeasureInclusion(alpha=0.09, osqp_warmstart=False) 

# alg_orig = OccupationMeasureInclusion(alpha=0.09, osqp_warmstart=False, osqp_atol=1e-3, osqp_rtol=1e-3)

### indeed checked that this is different from the first version above, namely without specifying osqp_atol and osqp_rtol? --> explained above
### also checked that warmstart True vs. False here does not change the plot in the eyeball checking sense, but change the pi a lot (False: -1500+ entries; True: -200+ entries; but both wrong/need fix; see above for the fix)
alg_orig = OccupationMeasureInclusion(
    alpha=0.09, osqp_warmstart=False, osqp_atol=1e-8, osqp_rtol=1e-8
)  

# note that actually atol=None, rtol=None corresponds to atol=rtol=0 based on _trigger_early_stopping
# so the first line below would indeed terminate earlier
# _, expls_orig, _ = alg_orig.solve(env, atol=1e-3, rtol=1e-3, verbose=True)
pis_orig, expls_orig, rts_orig = alg_orig.solve(env, atol=None, rtol=None, verbose=True)

# tune() returns an optuna.Study object
study = alg_orig.tune(metric=GeometricMean(shift=0.5), envs=[env], n_trials=80)

# which we can use to initialize a new instance
alg_tuned = alg_orig.from_study(study)

# _, expls_tuned, _ = alg_tuned.solve(env, atol=1e-3, rtol=1e-3, verbose=True)
pis_tuned, expls_tuned, rts_tuned = alg_tuned.solve(
    env, atol=None, rtol=None, verbose=True
)


plt.xlabel("Iteration")
plt.ylabel("Exploitability")
plt.plot(expls_orig, label="Original")
plt.plot(expls_tuned, label="Tuned")
plt.semilogy()
plt.grid()
plt.legend()
plt.show()

### sanity check it's not normalized
print("### sanity check expls not normalized")
print(expls_orig[0], expls_tuned[0])
print()

### sanity check constraint violation
print("### pis orig sum to one violation")
print([(pis_orig_i.sum(axis=-1) - 1).abs().max().item() for pis_orig_i in pis_orig])
print()
print("### pis orig nonnegativity violation")
print([pis_orig_i.min().item() for pis_orig_i in pis_orig])
print()
print("### pis tuned sum to one violation")
print([(pis_tuned_i.sum(axis=-1) - 1).abs().max().item() for pis_tuned_i in pis_tuned])
print()
print("### pis tuned nonnegativity violation")
print([pis_tuned_i.min().item() for pis_tuned_i in pis_tuned])


### [DONE] TODO: Check why tune not showing the optuna process printouts?
### Oh just because of ”optuna.logging.set_verbosity(optuna.logging.WARNING)“？

### [DONE] TODO: Btw the differences compared to the example.ipynb results of this example in 0089d3121c8dde5484ea8a1473728b100d30102e commit Add integration testing may be due to the enforce feasibility stuff on vs. off double check?
### No, found that the difference of tuned solve it's indeed just due to missing osqp_warmstart in _init_tuner_instance --> found this by adding "print(f"DEBUG: {osqp_rtol=}, {osqp_atol=}, {self.osqp_warmstart=}")"" in step_next_state
### The difference of original solve is simply due to atol=rtol=None vs. atol=rtol=1e-3? Seems that when I tried initially last week 20250812, the original one matched and so maybe I luckily didn't edit the atol=None, rtol=None for original solve there but only edited the tuned solve one.


### TODO: Check constraint violation? Seems that indeed 1e-3 osqp_atol/osqp_rtol leads to even lower constraint violations in general? Also osqp_atol/osqp_rtol do not seem to be really default to 1e-8 due to atol=rtol=None?
### Moreover, seems that the min of pi entries can be as negative as -1600+, etc. which does not make any sense regardless of whether 1e-3 or 1e-8 osqp tolerances? Also double check if sum up to one violations make sense in general or not.
### Maybe something is still missing in the osqp_atol and osqp_rtol logic? Double check. Mainly check and compare the following.
### Also don't forget about solve and solve_kwargs doc improvement: https://github.com/radar-research-lab/MFGLib/pull/55#issuecomment-3018458683
"""
# alg_orig = OccupationMeasureInclusion(alpha=0.09, osqp_warmstart=False) # this corresponds to 1e-8 osqp_atol and osqp_rtol? but eventual constraint violation seems to be even more based on the checking below?
# alg_orig = OccupationMeasureInclusion(alpha=0.09, osqp_warmstart=False, osqp_atol=1e-3, osqp_rtol=1e-3)
alg_orig = OccupationMeasureInclusion(alpha=0.09, osqp_warmstart=False, osqp_atol=1e-8, osqp_rtol=1e-8) # indeed checked that this is different from the first version above, namely without specifying osqp_atol and osqp_rtol?
"""
