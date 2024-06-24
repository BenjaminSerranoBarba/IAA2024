#!/usr/bin/env python
# coding: utf-8

import random as rnd
import numpy as np

class Problem:
    def __init__(self):
        self.dimension = 5
        self.clients = range(self.dimension) # [0,1,2,3,4]
        """self.costs = [
            [0,  5,  10, 15, 7,  12, 20, 8,  14, 10, 9],
            [5,  0,  6,  9,  5,  8,  14, 4,  7,  5,  6],
            [10, 6,  0,  8,  4,  6,  13, 7,  5,  8,  9],
            [15, 9,  8,  0,  7,  4,  10, 5,  3,  6,  12],
            [7,  5,  4,  7,  0,  3,  9,  6,  4,  5,  8],
            [12, 8,  6,  4,  3,  0,  7,  3,  5,  6,  9],
            [20, 14, 13, 10, 9,  7,  0,  6,  5,  8,  10],
            [8,  4,  7,  5,  6,  3,  6,  0,  2,  5,  7],
            [14, 7,  5,  3,  4,  5,  5,  2,  0,  3,  8],
            [10, 5,  8,  6,  5,  6,  8,  5,  3,  0,  4],
            [9,  6,  9,  12, 8,  9,  10, 7,  8,  4,  0]
        ] """
        """self.costs = [
            [0,  5,  10, 15, 7,  12, 20, 8,  14, 10, 9, 7],
            [5,  0,  6,  9,  5,  8,  14, 4,  7,  5,  6, 5],
            [10, 6,  0,  8,  4,  6,  13, 7,  5,  8,  9, 9],
            [15, 9,  8,  0,  7,  4,  10, 5,  3,  6,  12, 8],
            [7,  5,  4,  7,  0,  3,  9,  6,  4,  5,  8, 10],
            [12, 8,  6,  4,  3,  0,  7,  3,  5,  6,  9, 5],
            [20, 14, 13, 10, 9,  7,  0,  6,  5,  8,  10, 1],
            [8,  4,  7,  5,  6,  3,  6,  0,  2,  5,  7, 5],
            [14, 7,  5,  3,  4,  5,  5,  2,  0,  3,  8, 7],
            [10, 5,  8,  6,  5,  6,  8,  5,  3,  0,  4, 9],
            [9,  6,  9,  12, 8,  9,  10, 7,  8,  4,  0, 10],
            [2,  7,  11,  5, 4,  9,  6,  8,  1,  12, 15, 0]
        ] """
        self.costs = [
            [0, 10, 20, 15, 30],
            [10, 0, 25, 20, 10],
            [20, 25, 0, 30, 15],
            [15, 20, 30, 0, 18],
            [30, 10, 15, 18, 0]
        ]
        
        """self.costs = [
            [0, 10, 20, 15, 30, 20],
            [10, 0, 25, 20, 10, 15],
            [20, 25, 0, 30, 15, 6],
            [15, 20, 30, 0, 18, 7],
            [30, 10, 15, 18, 0, 8],
            [25, 8, 20, 16, 20, 0]
        ]"""

    def check(self, vec):
        # horizont
        # alidad-verticalidad
        table = []
        for i in self.clients:
            if vec[i] in table:
                return False
            table.append(vec[i])

        # ciclos
        visited = []
        next = self.clients[0] # == 0
        for _ in self.clients:
            pos = vec[next]
            if pos in visited:
                return False 
            visited.append(pos)
            next = pos
        return True

    def fit(self, vec):
        cost = 0
        for i in self.clients:
            cost += self.costs[i][vec[i]]
        return cost

class Vec: 
    def __init__(self):
        self.p = Problem()

        self.vec = [rnd.random() for _ in self.p.clients]
        self.discrete = [0] * self.p.dimension 
        self.discretize()
        self.updatelast()

    def isfeasible(self):
        return self.p.check(self.discrete)
    
    def isbetterthan(self, o):
        return self.fitness() < o.fitness()
    
    def fitness(self):
        return self.p.fit(self.discrete)
    
    def islastvecbetter(self):
        return self.fitnesslast() < self.fitness()
    
    def fitnesslast(self):
        return self.p.fit(self.discrete_last)
    
    def updatelast(self):
        self.vec_last = self.vec.copy()
        self.discrete_last = self.discrete.copy()

    def rollback(self):
        self.vec = self.vec_last.copy()
        self.discrete = self.discrete_last.copy()
    
    def move(self, F_best, F_medium, F_worst, Ft):
        for j in self.p.clients:
            Xbest = F_best.vec[j]
            Xmedium = F_medium.vec[j]
            Xworst = F_worst.vec[j]
            Xt = Xmedium - Xworst
            Xnew = (1.0 - Ft) * Xbest + rnd.random() * Ft * Xt
            self.vec[j] = Xnew

        self.discretize()    

    def move2(self, F_best, F_worst, ratio):
        for j in self.p.clients:
            Xbest = F_best.vec[j]
            Xworst = F_worst.vec[j]
            Xnew = self.vec[j] + rnd.random() * ratio * (Xbest - Xworst)
            self.vec[j] = Xnew

        self.discretize()

    def discretize(self):
        lista = sorted(self.vec)
        self.discrete = [lista.index(v) for v in self.vec]
    
    def __str__(self) -> str:
        return f"fit:{self.fitness()} x:{self.discrete}"

    def copy(self, a):
        self.vec = a.vec.copy()
        self.discrete = a.discrete.copy()
        self.vec_last = a.vec_last.copy()
        self.discrete_last = a.discrete_last.copy()

class GROM:
    GoldenRatio = (1 + (5.0 ** 0.5)) * 0.5

    def __init__(self):
        self.maxiter = 50
        self.nvecs = 5
        self.vectors = []
        self.g = Vec()
        assert self.nvecs % 2 == 1, "nvecs debe ser impar (cálculo de mediana)"
        
    def solve(self):
        self.initrand()
        self.evolve()
        print(self.g)

    def sublist(self, skiplist):
        sub = list()
        for pos in range(self.nvecs):
            if pos in skiplist:
                continue
            sub.append(self.vectors[pos])
        return sub

    def initrand(self):
        for i in range(self.nvecs):
            while True:
                p = Vec()
                if p.isfeasible():
                    break
            self.vectors.append(p)

        self.g.copy(self.vectors[0])
        for i in range(1, self.nvecs):
            if self.vectors[i].isbetterthan(self.g):
                self.g.copy(self.vectors[i])

    def evolve(self):
        t = 1
        while t <= self.maxiter:
            # 1st phase
            self.vectors.sort(key= lambda elem: elem.fitness())

            median_pos = self.nvecs // 2
            worst_pos = -1
            if self.vectors[median_pos].isbetterthan(self.vectors[worst_pos]):
                self.vectors[worst_pos].copy(self.vectors[median_pos])

            for i in range(self.nvecs):
                j = i
                while j == i:
                    j = rnd.randint(0, self.nvecs - 1)

                shortlist = self.sublist([i, j]) # list without i and j

                F = [self.vectors[i], self.vectors[j], shortlist[(self.nvecs - 2) // 2]] # i vec, j vec, median vec
                F.sort(key= lambda elem: elem.fitness())

                F_best = F[0]
                F_medium = F[1]
                F_worst = F[2]
                T = t / self.maxiter
                Ft = (GROM.GoldenRatio / (5**0.5)) * (GROM.GoldenRatio**T - (1 - GROM.GoldenRatio)**T)
                if np.iscomplex(Ft):
                    Ft = Ft.real

                while True:
                    vec = Vec()
                    vec.copy(self.vectors[i])
                    vec.move(F_best, F_medium, F_worst, Ft)
                    #print(f"iter {t}phase1 discrete[{i}] = {vec.discrete}")
                    print("1°", self.g)
                    if vec.isfeasible():
                        self.vectors[i].copy(vec)
                        break

                if t > 1 and self.vectors[i].islastvecbetter():
                    self.vectors[i].rollback()
                else:
                    self.vectors[i].updatelast()
            
            # 2nd phase
            self.vectors.sort(key= lambda elem: elem.fitness())
            for i in range(self.nvecs):
                F_best = self.vectors[0]
                F_worst = self.vectors[-1]
                ratio = 1.0 / GROM.GoldenRatio
                while True:
                    vec = Vec()
                    vec.copy(self.vectors[i])
                    vec.move2(F_best, F_worst, ratio)
                    #print(f"iter {t}phase2 discrete[{i}] = {vec.discrete}")
                    print("2°", self.g)
                    if vec.isfeasible():
                        self.vectors[i].copy(vec)
                        break

                if t > 1 and self.vectors[i].islastvecbetter():
                    self.vectors[i].rollback()
                else:
                    self.vectors[i].updatelast()
                    
                if self.vectors[i].isbetterthan(self.g):
                    self.g.copy(self.vectors[i])
            t += 1

#try:
GROM().solve()
#except Exception as e:
#    print(f"{e} \nCaused by {e.__cause__}")
