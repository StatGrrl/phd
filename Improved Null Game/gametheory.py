"""
Module docstring
"""


# Subgame Perfect Nash Equilibrium for a 2-player Stackelberg extensive form game
def spne(c1, c2, u1, u2, leader):
    """
    Given the strategies and utilities for 2 players, return the unique
    Subgame Perfect Nash Equilibrium (SPNE), found by backward induction,
    when the leader moves first.
    Payoffs must be unique for each player in order to guarantee a unique SPNE.

    Parameters
    ----------
    c1 : list of strings
      The set of strategies for player 1
    u1 : array of integers or floats
      The utility matrix for player 1, payoffs must be unique
    c2 : list of strings
      The set of strategies for player 2
    u2 : array of integers or floats
      The utility matrix for player 2, payoffs must be unique
    leader: string, 'p1' for player 1 or 'p2' for player 2
        The player which moves first

    Returns
    -------
    list containing SPNE strategies and corresponding payoffs

    Examples
    --------

    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p1')
    {'p1_optimal': 'b', 'p2_optimal': 'd', 'p1_utility': 3, 'p2_utility': 5}

    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p2')
    {'p1_optimal': 'c', 'p2_optimal': 'd', 'p1_utility': 4, 'p2_utility': 3}

    """

    import numpy as np

    len1 = len(c1)
    len2 = len(c2)
    range1 = list(range(len1))
    range2 = list(range(len2))
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
        strategies = [c1[p1_best_resp], c2[p2_best_resp]]
        payoff = [u1[p1_best_resp][p2_best_resp], u2[p1_best_resp][p2_best_resp]]

    # Player 2 Leads
    else:
        u1temp = list(zip(*u1))
        u2temp = list(zip(*u2))
        for i in range2:
            p1_best_resp.append(np.argmax(u1temp[i]))
        for i in range2:
            p2_best_resp.append(u2temp[i][p1_best_resp[i]])
        p2_best_resp = int(np.argmax(p2_best_resp))
        p1_best_resp = p1_best_resp[p2_best_resp]

    return {'p1_optimal': c1[p1_best_resp], 'p2_optimal': c2[p2_best_resp],
            'p1_utility': u1[p1_best_resp][p2_best_resp], 'p2_utility': u2[p1_best_resp][p2_best_resp]}
