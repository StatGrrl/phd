"""
Module docstring
"""
import numpy as np

# Convert 1d or 2d array to dictionary
def array_to_dict(arr):
    """ Return arr in dictionary form, where arr is 1d or 2d list (faster) or numpy array"""
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    d = dict()
    if isinstance(arr[0], list):
        for x, row in enumerate(arr):
            for y, element in enumerate(row):
                d[(x, y)] = element
    else:
        for x, element in enumerate(arr):
            d[x] = element
    return d

# Expected Utility
def exp_util(x1, u1, x2, u2, leader):
    """
    Given the mixed strategy and utility matrix for each player, calculate expected utility for Player 1 and Player 2

    Parameters
    ----------
    x1 : 1d numpy array of floats - the mixed strategy for player 1
    u1 : 2d numpy array of floats - the utility matrix for player 1
    x2 : 1d numpy array of floats - the mixed strategy for player 2
    u2 : 2d numpy array of floats - the utility matrix for player 2

    Returns
    -------
    p1_strategies, p1_payoffs, p2_strategies, p2_payoffs

    Example
    -------
    >>> exp_util(np.array([0, 0.2, 0.8]), np.array([[1, 0], [3, 5], [4, 2]], np.float64), \
                 np.array([0.75, 0.25]), np.array([[0, 2], [5, 1], [3, 4]], np.float64))
    (3.5, 3.4000000000000004)

    >>> exp_util(np.array([0.75, 0.25]), np.array([[1, 3, 4], [0, 5, 2]], np.float64), \
                 np.array([0, 0.2, 0.8]), np.array([[0, 5, 3], [2, 1, 4]], np.float64))
    (3.5000000000000004, 3.4000000000000004)

    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    u1 = np.array(u1)
    u2 = np.array(u2)
    
    if x1.ndim == 1:
        x1 = x1.reshape(len(x1),1)
    if x2.ndim == 1:
        x2 = x2.reshape(len(x2),1)
    
    if leader == 'ranger':
        p1_payoff = np.matmul(np.matmul(np.transpose(x1), u1), x2)[0][0]
        p2_payoff = np.matmul(np.matmul(np.transpose(x1), u2), x2)[0][0]
    if leader == 'poacher':
        p1_payoff = np.matmul(np.matmul(np.transpose(x2), u1), x1)[0][0]
        p2_payoff = np.matmul(np.matmul(np.transpose(x2), u2), x1)[0][0]

    return x1, p1_payoff, x2, p2_payoff

# Subgame Perfect Nash Equilibrium for a 2-player extensive form game
def spne(c1, c2, u1, u2, leader):
    """
    Given the strategies and utilities for 2 players, return the unique
    Subgame Perfect Nash Equilibrium (SPNE), found by backward induction,
    when the leader moves first.
    Payoffs must be unique for each player in order to guarentee a unique SPNE - No tie handling / mixed strategy

    Examples
    --------

    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p1')
    (['b', 'd'], [3, 5])

    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p2')
    (['c', 'd'], [4, 3])

    """

    len1 = len(c1)
    len2 = len(c2)
    range1 = range(len1)
    range2 = range(len2)
    p1_best_resp = []
    p2_best_resp = []

    # Player 1 Leads
    if leader == 'p1':
        for i in range1:
            p2_best_resp.append(np.argmax(u2[i]))
        for i in range1:
            p1_best_resp.append(u1[i][p2_best_resp[i]])
        p1_best_resp = int(np.argmax(p1_best_resp))
        p2_best_resp = int(p2_best_resp[p1_best_resp])

    # Player 2 Leads
    if leader == 'p2':
        u1temp = list(zip(*u1))
        u2temp = list(zip(*u2))
        for i in range2:
            p1_best_resp.append(np.argmax(u1temp[i]))
        for i in range2:
            p2_best_resp.append(u2temp[i][p1_best_resp[i]])
        p2_best_resp = int(np.argmax(p2_best_resp))
        p1_best_resp = p1_best_resp[p2_best_resp]

    strat = [c1[p1_best_resp], c2[p2_best_resp]]
    payoff = [u1[p1_best_resp][p2_best_resp], u2[p1_best_resp][p2_best_resp]]

    return strat, payoff

# nash using gambit
def nash(c1, c2, u1, u2):
    import pygambit
    g = pygambit.Game.new_table([len(c1), len(c2)])
    g.title = 'GAME12'
    g.players[0].label = 'ranger'
    g.players[1].label = 'poacher'
    dim = u1.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            g[i, j][0] = int(u1[i, j])
            g[i, j][1] = int(u2[i, j])
    eq = pygambit.nash.lcp_solve(g, use_strategic=True, rational=False)
    p1_strat = eq[0][g.players[0]] 
    p2_strat = eq[0][g.players[1]]
    # p1_payoff = eq[0].payoff(g.players[0]) # incorrect since payoffs int
    # p2_payoff = eq[0].payoff(g.players[1])
    util = exp_util(p1_strat, u1, p2_strat, u2, 'ranger')
    return p1_strat, util[1], p2_strat, util[3], len(eq)

# maximin
def maximin(u, player='ranger'):
    import pulp as plp

    if player == 'poacher':
        u = np.transpose(u)

    n = u.shape[0]
    m = u.shape[1]
    set_i = range(n)
    set_j = range(m)
    u1_list = u.tolist()
    u1_input = array_to_dict(u1_list)

    # formulate LP and slve
    x_vars = dict()
    for i in set_i:
        x_vars[i] = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=1, name=str(i).zfill(4))
    
    rho_var = plp.LpVariable(cat=plp.LpContinuous, name="rho")

    opt_model = plp.LpProblem(name="Maximin")
    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] for i in set_i), sense=plp.LpConstraintEQ,
                                             rhs=1, name="constraint"))
    constraints2 = {j: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] * u1_input[i, j] for i in set_i) - rho_var,
                                                                sense=plp.LpConstraintGE, rhs=0,
                                                                name="constraints2_{0}".format(j)))
                    for j in set_j}

    objective = rho_var
    opt_model.sense = plp.LpMaximize
    opt_model.setObjective(objective)
    opt_model.solve() 
    
    # get optimal strategy
    x = np.zeros(n)
    for var in opt_model.variables():
        if var.name == 'rho':
            rho = var.varValue
        for i in set_i:
            if var.name == str(i).zfill(4):
                x[i] = var.varValue
    
    feasible = opt_model.status == 1
    optimal = np.round(opt_model.objective.value(), 3) == round(rho, 3)

    return x, rho, feasible, optimal

def minimax(u, player='ranger'):
    import pulp as plp

    if player == 'poacher':
        u = np.transpose(u)

    n = u.shape[0]
    m = u.shape[1]
    set_i = range(n)
    set_j = range(m)
    u1_list = u.tolist()
    u1_input = array_to_dict(u1_list)

    # formulate LP and slve
    x_vars = dict()
    for i in set_i:
        x_vars[i] = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=1, name=str(i).zfill(4))
    
    rho_var = plp.LpVariable(cat=plp.LpContinuous, name="rho")

    opt_model = plp.LpProblem(name="Minimax")
    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] for i in set_i), sense=plp.LpConstraintEQ,
                                             rhs=1, name="constraint"))
    constraints2 = {j: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] * u1_input[i, j] for i in set_i)-rho_var,
                                                                sense=plp.LpConstraintLE, rhs=0,
                                                                name="constraints2_{0}".format(j)))
                    for j in set_j}

    objective = rho_var
    opt_model.sense = plp.LpMinimize
    opt_model.setObjective(objective)
    opt_model.solve() 
    
    # get optimal strategy
    x = np.zeros(n)
    for var in opt_model.variables():
        if var.name == 'rho':
            rho = var.varValue
        for i in set_i:
            if var.name == str(i).zfill(4):
                x[i] = var.varValue
    
    feasible = opt_model.status == 1
    optimal = np.round(opt_model.objective.value(), 3) == round(rho, 3)

    return x, rho, feasible, optimal


# SSG Follower Problem - Linear Program (Paruchuri et al(2008))
def ssg_follower(x1, u2, low=0, up=1, leader='poacher'):
    """
    Given the mixed strategy for Player 1 (leader) and the utility matrix for Player 2 (follower),
    return the optimal mixed strategy and expected payoff for Player 2 (follower)

    Parameters
    ----------
    x1 : 1d numpy array of floats - the mixed strategy for player 1 (leader)
    u2 : 2d numpy array of floats - the utility matrix for player 2 (follower)

    Returns
    -------
    p2_strategy, p2_payoff

    Example
    -------
    >>> ssg_follower(np.array([0, 0.2, 0.8]), np.array([[0, 2], [5, 1], [3, 4]], np.float64))
    (array([1., 0.]), 3.4000000000000004)

    >>> ssg_follower(np.array([0.75, 0.25]), np.array([[0, 5, 3], [2, 1, 4]], np.float64))
    (array([0., 1., 0.]), 4.0)

    """

    import pulp as plp

    if leader == 'ranger':
        u2 = np.transpose(u2)
    
    m = u2.shape[0]
    n = u2.shape[1]
    set_i = range(m)
    set_j = range(n)
    x1_list = x1.tolist()
    u2_list = u2.tolist()
    x1_input = array_to_dict(x1_list)
    u2_input = array_to_dict(u2_list)

    # optimal value
    util = np.matmul(u2, x1)
    max_util = util.max()

    # get support and calculate upper bound
    if up == 1:
        support = np.where(np.round(util, 3) == np.round(max_util, 3))
        up = 1 / len(support[0])

    # formulate LP and slve
    x2_vars = dict()
    for i in set_i:
        x2_vars[i] = plp.LpVariable(cat=plp.LpContinuous, lowBound=low, upBound=up, name=str(i).zfill(4))

    opt_model = plp.LpProblem(name="Follower_Problem")
    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x2_vars[i] for i in set_i), sense=plp.LpConstraintEQ,
                                             rhs=1, name="constraint"))
    objective = plp.lpSum(x2_vars[i] * u2_input[i, j] * x1_input[j] for i in set_i for j in set_j)
    opt_model.sense = plp.LpMaximize
    opt_model.setObjective(objective)
    opt_model.solve() 
    
    # get optimal strategy
    x2_output = np.zeros(m)
    for i in set_i:
        x2_output[i] = opt_model.variables()[i].varValue
    
    feasible = opt_model.status == 1
    optimal = np.round(opt_model.objective.value(), 3) == np.round(max_util, 3)

    return x2_output, opt_model.objective.value(), feasible, optimal, util

# SSG Leader Problem - Mixed integer Linear Program (DOBSS - Paruchuri et al (2008)) but only 1 poacher type
def dobss(u1, u2, m_const=1000):
    """
    Given the utility matrices for Player 1 (leader) and Player 2 (follower),
    return the optimal mixed strategy and expected payoff for Player 1 (leader) and
    the optimal pure strategy and expected payoff for Player 2 (follower)

    Parameters
    ----------
    u1 : 2d numpy array of floats - the utility matrix for player 1 (leader)
    u2 : 2d numpy array of floats - the utility matrix for player 2 (follower)
    m_const: some large constant

    Returns
    -------
    p1_mixed_strategy, p1_payoff, p2_pure_strategy, p2_payoff

    Example
    -------
    >>> dobss(np.array([[1, 0], [3, 5], [4, 2]], np.float64), np.array([[0, 2], [5, 1], [3, 4]], np.float64))
    (array([0. , 0.2, 0.8]), 3.8000000000000003, array([1., 0.]), 3.4)

    >>> dobss(np.array([[1, 3, 4], [0, 5, 2]], np.float64), np.array([[0, 5, 3], [2, 1, 4]], np.float64))
    (array([0.6, 0.4]), 3.8, array([0., 1., 0.]), 3.4)

    """

    import pulp as plp

    n = u1.shape[0]
    m = u2.shape[1]
    set_i = range(n)
    set_j = range(m)
    u1_list = u1.tolist()
    u2_list = u2.tolist()
    u1_input = array_to_dict(u1_list)
    u2_input = array_to_dict(u2_list)

    z_vars = dict()
    for i in set_i:
        for j in set_j:
            z_vars[i, j] = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=1, name="z_{0}_{1}".format(i, j))

    q_vars = dict()
    for j in set_j:
        q_vars[j] = plp.LpVariable(cat=plp.LpBinary, name="q_{0}".format(j))

    a_var = plp.LpVariable(cat=plp.LpContinuous, name="a")

    opt_model = plp.LpProblem(name="DOBSS")

    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(z_vars[i, j] for i in set_i for j in set_j),
                                             sense=plp.LpConstraintEQ, rhs=1, name="constraint1"))

    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(q_vars[j] for j in set_j),
                                             sense=plp.LpConstraintEQ, rhs=1, name="constraint2"))

    constraints3 = {i: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(z_vars[i, j] for j in set_j),
                                                                sense=plp.LpConstraintLE, rhs=1,
                                                                name="constraints3_{0}".format(i)))
                    for i in set_i}

    constraints4 = {j: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(z_vars[i, j] for i in set_i),
                                                                sense=plp.LpConstraintLE, rhs=1,
                                                                name="constraints4_{0}".format(j)))
                    for j in set_j}

    constraints5 = {j: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(z_vars[i, j] for i in set_i) - q_vars[j],
                                                                sense=plp.LpConstraintGE, rhs=0,
                                                                name="constraints5_{0}".format(j)))
                    for j in set_j}

    constraints6 = {j: opt_model.addConstraint(
        plp.LpConstraint(e=a_var - plp.lpSum(u2_input[i, j] * plp.lpSum(z_vars[i, h] for h in set_j) for i in set_i) -
                         (1 - q_vars[j]) * m_const,
                         sense=plp.LpConstraintLE, rhs=0,
                         name="constraints6_{0}".format(j)))
                    for j in set_j}

    constraints7 = {j: opt_model.addConstraint(
        plp.LpConstraint(e=a_var - plp.lpSum(u2_input[i, j] * plp.lpSum(z_vars[i, h] for h in set_j) for i in set_i),
                         sense=plp.LpConstraintGE, rhs=0, name="constraints7_{0}".format(j)))
                    for j in set_j}

    objective = plp.lpSum(u1_input[i, j] * z_vars[i, j] for i in set_i for j in set_j)

    opt_model.sense = plp.LpMaximize
    opt_model.setObjective(objective)

    # solving with CBC
    opt_model.solve()

    q = np.zeros(m)
    z = np.zeros((n, m))
    a = None
    pure_idx = None
    for var in opt_model.variables():
        if var.name == 'a':
            a = var.varValue
        for j in set_j:
            if var.name == ('q_' + str(j)):
                q[j] = var.varValue
                if q[j] == 1.0:
                    pure_idx = j
            for i in set_i:
                if var.name == 'z_' + str(i) + '_' + str(j):
                    z[i, j] = var.varValue
    x = np.zeros(n)
    for i in set_i:
        x[i] = z[i, pure_idx]

    return x, opt_model.objective.value(), q, a

# Conitzer 2006 Leader LP
def leadership(u1, u2):
    import pulp as plp

    n = u1.shape[0]
    m = u2.shape[1]
    set_i = range(n)
    set_j = range(m)
    u1_list = u1.tolist()
    u2_list = u2.tolist()
    u1_input = array_to_dict(u1_list)
    u2_input = array_to_dict(u2_list)

    x_vars = dict()
    for i in set_i:
        x_vars[i] = plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=1, name="x_{0}".format(i))

    opt_model = plp.LpProblem(name="Leadership")
    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] for i in set_i),
                                             sense=plp.LpConstraintEQ, rhs=1, name="constraint1"))
    constraints2 = {k: opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(x_vars[i] * u2_input[i,j] - x_vars[i] * u2_input[i,k] for i in set_i for j in set_j),
                                                                sense=plp.LpConstraintGE, rhs=0,
                                                                name="constraints2_{0}".format(k)))
                    for k in set_j}

    objective = plp.lpSum(u1_input[i, j] * x_vars[i] for i in set_i for j in set_j)
    opt_model.sense = plp.LpMaximize
    opt_model.setObjective(objective)
    opt_model.solve()

    x = np.zeros(n)
    for var in opt_model.variables():
        for i in set_i:
            if var.name == ('x_' + str(i)):
                x[i] = var.varValue
    return x, opt_model.objective.value()
