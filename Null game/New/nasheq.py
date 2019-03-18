# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:40:28 2015

@author: Lisa
"""

# Nash Equilibria for a 2-player strategic game
def nash(c1, c2, u1, u2):
    """
    Given the strategies and utilities for 2 players, return all Nash
    equilibria when players move simultaneously.

    Parameters
    ----------
    c1 : list of strings
      The set of strategies for player 1
    u1 : list of numbers
      The utility matrix for player 1
    c2 : list of strings
      The set of strategies for player 2
    u2 : list of numbers
      The utility matrix for player 2

    Returns
    -------
    list containing Nash equilibria strategies and corresponding payoffs 

    Example
    -------
    >>> nash(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]])
    ([[array([ 0. ,  0.2,  0.8]), array([ 0.75,  0.25])]], [[3.5, 3.3999999999999999]])

    """

    import itertools
    import numpy

    # Pure Strategy Nash Equilibria
    len1 = len(c1)
    len2 = len(c2)
    range1 = range(len1)
    range2 = range(len2)
    purestrat = []
    purepay = []
    for i in range1:
        range1temp = range(len1)
        range1temp.remove(i)
        for j in range2:
            temp = 1
            for k in range1temp:
                temp = temp * int(u1[i][j] > u1[k][j])
            if temp == 1:
                range2temp = range(len2)
                range2temp.remove(j)
                for k in range2temp:
                    temp = temp * int(u2[i][j] > u2[i][k])
            if temp == 1:
                purestrat.append([c1[i], c2[j]])
                purepay.append([u1[i][j], u2[i][j]])

    # Randomized Strategy Nash Equilibria
    supp1 = []
    for i in range1:
        if i != 0:
            for j in itertools.combinations(range1, i+1):
                supp1.append(j)
    supp2 = []
    for i in range2:
        if i != 0:
            for j in itertools.combinations(range2, i+1):
                supp2.append(j)
    randstrat = []
    randpay = []
    for i in range(len(supp1)):
        for j in range(len(supp2)):
            lensupp1 = len(supp1[i])
            lensupp2 = len(supp2[j])
            dim = lensupp1 + lensupp2 + 2
            linsys = numpy.zeros(shape=(dim, dim))
            col1 = 0
            row2 = lensupp2
            for k in supp1[i]:
                row1 = 0
                col2 = lensupp1
                for l in supp2[j]:
                    linsys[row1][col1] = u2[k][l]
                    linsys[row2][col2] = u1[k][l]
                    linsys[row1][dim - 1] = -1
                    linsys[row2][dim - 2] = -1
                    linsys[dim - 2, col1] = 1
                    linsys[dim - 1, col2] = 1
                    row1 += 1
                    col2 += 1
                col1 += 1
                row2 += 1
            linaug = numpy.zeros(shape=dim)
            linaug[dim - 2] = 1
            linaug[dim - 1] = 1
            temp = 0
            if numpy.linalg.det(linsys) <> 0:
                linsol = numpy.linalg.solve(linsys, linaug)            
                if all(linsol[:dim - 2] > 0):
                    temp = 1
                    strat1 = []
                    strat2 = []
                    rand1 = numpy.zeros(shape=len1)
                    rand2 = numpy.zeros(shape=len2)
                    for k in range(lensupp1):
                        strat1.append(c1[supp1[i][k]])
                    for k in range(lensupp2):
                        strat2.append(c2[supp2[j][k]])
                    rand1temp = linsol[0: lensupp1]
                    rand2temp = linsol[lensupp1: lensupp1 + lensupp2]
                    rand1[list(supp1[i])] = rand1temp
                    rand2[list(supp2[j])] = rand2temp 
                    pay1 = linsol[dim - 2]
                    pay2 = linsol[dim - 1]
                    for k in range1:
                        if strat1.count(c1[k]) == 0:
                            pay = 0
                            for l in range(lensupp2):
                                pay += rand2temp[l] * u1[k][l]
                            if pay > pay1:
                                temp = temp * 0
                    for k in range2:
                        if strat2.count(c2[k]) == 0:
                            pay = 0
                            for l in range(lensupp1):
                                pay += rand1temp[l] * u2[l][k]
                            if pay > pay2:
                                temp = temp * 0
                    if temp == 1:
                        randpay.append([pay1, pay2])
                        randstrat.append([rand1, rand2])
    strat = purestrat + randstrat
    payoff = purepay + randpay
    return strat, payoff

# Subgame Perfect Nash Equilibrium for a 2-player extensive form game 
def spne(c1, c2, u1, u2, leader):
    """
    Given the strategies and utilities for 2 players, return the unique 
    Subgame Perfect Nash Equilibrium (SPNE), found by backward induction, 
    when the leader moves first. 
    Payoffs must be unique for each player in order to guarentee a unique SPNE.

    Parameters
    ----------
    c1 : list of strings
      The set of strategies for player 1
    u1 : list of numbers
      The utility matrix for player 1, payoffs must be unique
    c2 : list of strings
      The set of strategies for player 2
    u2 : list of numbers
      The utility matrix for player 2, payoffs must be unique
    leader: string, 'p1' for player 1 or 'p2' for player 2
        The player which moves first

    Returns
    -------
    list containing SPNE strategies and corresponding payoffs 

    Examples
    --------
    
    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p1')
    (['b', 'd'], [3, 5])
    
    >>> spne(['a', 'b', 'c'], ['d', 'e'], [[1, 0], [3, 5], [4, 2]], [[0, 2], [5, 1], [3, 4]], 'p2')
    (['c', 'd'], [4, 3])

    """

    import numpy

    len1 = len(c1)
    len2 = len(c2)
    range1 = range(len1)
    range2 = range(len2)
    p1_bestresp = []
    p2_bestresp = []

    # Player 1 Leads
    if leader == 'p1':
        for i in range1:
            p2_bestresp.append(numpy.argmax(u2[i]))
        for i in range1:
            p1_bestresp.append(u1[i][p2_bestresp[i]])
        p1_bestresp = numpy.argmax(p1_bestresp)
        p2_bestresp = p2_bestresp[p1_bestresp]
        strat = [c1[p1_bestresp], c2[p2_bestresp]]
        payoff = [u1[p1_bestresp][p2_bestresp], u2[p1_bestresp][p2_bestresp]]
    
    # Player 2 Leads
    if leader == 'p2':
        u1temp = zip(*u1)
        u2temp = zip(*u2)
        for i in range2:
            p1_bestresp.append(numpy.argmax(u1temp[i]))
        for i in range2:
            p2_bestresp.append(u2temp[i][p1_bestresp[i]])
        p2_bestresp = numpy.argmax(p2_bestresp)
        p1_bestresp = p1_bestresp[p2_bestresp]
        strat = [c1[p1_bestresp], c2[p2_bestresp]]
        payoff = [u1[p1_bestresp][p2_bestresp], u2[p1_bestresp][p2_bestresp]]
    
    return strat, payoff


# Examples
    
# myerson1997
c1 = ['x1', 'y1']
c2 = ['x2', 'y2']
u1 = [[2, 3], [1, 4]]
u2 = [[1, 2], [4, 3]]
nash(c1, c2, u1, u2)
spne(c1, c2, u1, u2, 'p1')
spne(c1, c2, u1, u2, 'p2')

# an2015
c1 = ['a', 'b']
c2 = ['c', 'd']
u1 = [[3, 5], [2, 4]]
u2 = [[1, 0], [0, 2]]
nash(c1, c2, u1, u2)
spne(c1, c2, u1, u2, 'p1')
spne(c1, c2, u1, u2, 'p2')
