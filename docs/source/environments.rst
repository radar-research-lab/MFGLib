Environments
============

Built-In
--------

MFGLib comes with 10 pre-implemented environments which can be accessed by calling the corresponding classmethods
of ``Environment``. The pre-implemented environments are listed below:

.. automethod:: mfglib.env::Environment.beach_bar

.. automethod:: mfglib.env::Environment.building_evacuation

.. automethod:: mfglib.env::Environment.conservative_treasure_hunting

.. automethod:: mfglib.env::Environment.crowd_motion

.. automethod:: mfglib.env::Environment.equilibrium_price

.. automethod:: mfglib.env::Environment.left_right

.. automethod:: mfglib.env::Environment.linear_quadratic

.. automethod:: mfglib.env::Environment.random_linear

.. automethod:: mfglib.env::Environment.rock_paper_scissors

.. automethod:: mfglib.env::Environment.susceptible_infected

All implemented algorithms are parameterized so that you can control the size of the state space, action space, and time
horizon. In the following example, we create two distinct buildings, one with 10 floors each 20 by 20, and another with
100 floors each 50 by 5.

.. code-block:: python

   from mfglib.env import Environment
   
   env_1 = Environment.building_evacuation(n_floor=10, floor_l=20, floor_w=20)
   env_2 = Environment.building_evacuation(n_floor=100, floor_l=50, floor_w=5)


User-Defined
------------

Any environment defined in this library has the following attributes:

* ``T``: Sets the time horizon of the environment from 0 to ``T`` (inclusive, integer steps).
* ``S``: State space shape. For example, if the state space is all the integers from 1 to 100, then ``S=(100,)``, and if the state space is all the integer grid points :math:`(x, y)` such that :math:`1 \leq x,y \leq 100`, then ``S=(100, 100)``.
* ``A``: Action space shape.
* ``mu0``: Initial state distribution.
* ``r_max``: The supremum of the absolute value of rewards. This parameter is only used in **Mean-Field Occupation Measure Optimization** algorithm and does not necessarily need to be exact. Even a loose upper bound would be sufficient.
* ``reward_fn``: Defines the reward function.
* ``transition_fn``: Defines the tranistion probability function.

.. note::
    Notice that in the integer grid points case, we could flatten the state space and show it using a one dimensional
    vector of size 10,000. But keeping the state (and action) space multi-dimensional, whenever it is possible, is the
    convention used in this library. This convention results in easier to interpret policies, mean-fields, rewards, etc.

**Policy and Mean-Field Tensors.** Given ``T``, ``S``, and ``A``, the shape of policy and mean-field tensors will be
``(T+1,) + S + A``. For example, if ``T=10, S=(20, 20), A=(5,)``, the policy and mean-field tensors will be of size
``(11, 20, 20, 5)``. In general, let ``S=(S_1, S_2, ..., S_n)`` and ``A=(A_1, A_2, ..., A_m)``, and let ``pi`` and
``L`` be a policy and a mean-field tensor, respectively. Then, ``pi[t, s_1, s_2, ..., s_n, a_1, a_2, ..., a_m]`` is
the probability of choosing action ``a = (a_1, a_2, ..., a_m)`` conditional on being at the state
``s = (s_1, s_2, ..., s_n)`` at time ``t``, and ``L[t, s_1, s_2, ..., s_n, a_1, a_2, ..., a_m]`` is the portion of
players that are in state ``s = (s_1, s_2, ..., s_n)`` and choose action ``a = (a_1, a_2, ..., a_m)`` at time ``t``.

**Reward Function.** We define the reward function via the argument ``reward_fn``. The user is allowed to pass either
a function or a class implementing ``__call__``. The inputs of the reward function must be ``env`` (an environment
instance), ``t`` (a specific time less than or equal to the time horizon), and ``L_t`` (the mean-field tensor at time
``t``). The output will be a tensor of shape ``S + A``. Let ``r`` be the output tensor, and assume
``S=(S_1, S_2, ..., S_n)`` and ``A=(A_1, A_2, ..., A_m)``. Then, ``r[s_1, s_2, ..., s_n, a_1, a_2, ..., a_m]`` is the
reward that agent gets from choosing action ``a=(a_1, a_2, ..., a_m)`` conditional on being at state
``s = (s_1, s_2, ..., s_n)``.

**Transition Function.** We define the transition probability function via the argument ``transition_fn``. The user is
allowed to pass either a function or a class implementing ``__call__``. The inputs of the transition probability
function must be ``env`` (an environment instance), ``t`` (a specific time less than or equal to the time horizon),
and ``L_t`` (the mean-field tensor at time ``t``). The output will be a tensor of shape ``S + S + A``. Let ``p`` be the
output tensor, and assume ``S=(S_1, S_2, ..., S_n)`` and ``A=(A_1, A_2, ..., A_m)``. Then,
``p[s2_1, s2_2, ..., s2_n, s1_1, s1_2, ..., s1_n, a_1, a_2, ..., a_m]`` is the probability of going to the state
``s2 = (s2_1, s2_2, ..., s2_n)`` conditional on being at the state ``s1 = (s1_1, s1_2, ..., s1_n)`` and choosing the
action ``a=(a_1, a_2, ..., a_m)``.

Custom Environment Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create a custom environment, you can define each one of the above-mentioned attributes and pass them to
``Environment``. Let's take a look at the environment **Random Linear**, which is a custom environment already
implemented in the library.

We first define the states and actions. We want to have ``n`` states and ``n`` actions. Therefore, ``S=(n,)`` and
``A=(n,)``. Also, we use a uniform initial state distribution. To get a specific instance, we consider ``n=5``.

.. code-block:: python

    import torch

    # Define the state and action space shape
    n = 5
    S = (n,)
    A = (n,)

    # Initial state distribution
    mu0 = torch.ones(n) / n

Now, we define the reward and transition functions. As the name of the environment suggests, we want the reward and
transition probabilities to be a random linear (affine indeed) function of the mean-field, that is given the mean
field :math:`L`, the reward and transition probabilities should be equal to :math:`M_1 \times L + M_2` for some
randomly generated matrices :math:`M_1, M_2`. We generate different pairs of matrices for reward and transition
functions.

Note that in order for transition probabilities to be well-defined, we apply a softmax function to the output of the
affine function. Furthermore, we restrict all the entries of the randomly generated matrices to be in :math:`[-m, m]`. 
With this constraint, it is fairly straightforward to see that the
absolute value of rewards cannot be larger than :math:`2m` implying that we should set ``r_max`` equal to :math:`2m`.
To get an environment instance, we set ``m=1``.  Putting it all together,

.. code-block:: python

    from mfg.env import Environment
    import torch

    n = 5
    m = 1

    torch.manual_seed(0)
    soft_max = torch.nn.Softmax(dim=-1)

    r1 = 2 * m * torch.rand(n, n) - m  # M_1 for reward_fn
    r2 = 2 * m * torch.rand(n, n) - m  # M_2 for reward_fn

    p1 = 2 * m * torch.rand(n, n, n) - m  # M_1 for transition_fn
    p2 = 2 * m * torch.rand(n, n, n) - m  # M_2 for transition_fn

    user_defined_random_linear = Environment(
        T=4,
        S=(n,),
        A=(n,),
        mu0=torch.ones(n) / n,
        r_max=2 * m,
        reward_fn=lambda env, t, L_t: r1 @ L_t + r2,
        transition_fn=lambda env, t, L_t: softmax(p1 @ L_t + p2),
    )

Refer to the MFGLib implementation of **Random Linear** for an alternative class-based implementation.
