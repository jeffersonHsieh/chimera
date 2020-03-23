import numpy as np
from tqdm import tqdm
from heapq import heapify, heappushpop

from data.reader import DataReader
from planner.planner import Planner
from scorer.scorer import Scorer
from utils.graph import Graph

import sys
if len(sys.argv) < 2:
    parallel = False
elif sys.argv[1] == "parallel":
    parallel = True

class NaivePlanner(Planner):
    is_parallel = parallel
    re_plan = "PREMADE"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        for i in range(5):
            self.scorer.learn(train_reader, dev_reader)
            if not self.scorer.is_trainable:
                break
        return self

    def score(self, g: Graph, plan: str):
        return self.scorer.score(plan)
    
    def plan_best_unpack(self, args):
        g, low_mem = args
        return self.plan_best(g, low_mem = low_mem)

    def plan_best(self, g: Graph, ranker_plans=None, low_mem = False):
        if low_mem:
            if ranker_plans:
                all_plans = list(set(ranker_plans))
            else:
                all_plans = self.plan_all_o1(g)
            #exit(0)
            #best_50_plans = []
            best_plan = []
            best_score = 0 
            count = 0
            for p in all_plans:
                #print(p)
                count += 1
                pl = p.split()
                pl = [ c for c in pl if c]
                if pl[0] == ".":
                    pl.pop(0)
                p = " ".join(pl)
                pdot = p.split(".")
                pdot = [c for c in pdot if (c and c !=" ")]
                p = ".".join(pdot)
                score = self.scorer.score(p)
                if score > best_score:
                    if best_plan:
                        best_plan.pop()
                    best_plan.append(p)
                '''if len(best_50_plans)< 50:
                    best_50_plans.append((score,p))
                elif len(best_50_plans) == 50:
                    heapify(best_50_plans)
                    heappushpop(best_50_plans, (score,p))
                else:
                    heappushpop(best_50_plans ,(score,p))'''
            #print(best_plan)
            #best_50_plans = [p for s,p in best_50_plans]

            print(count)
            #print(best_50_plans)
            return best_plan
        if ranker_plans:
            all_plans = list(set(ranker_plans))
        else:
            all_plans = list(self.plan_all(g))
        plan_scores = [(p, self.scorer.score(p)) for p in tqdm(all_plans)]
        plan_scores = sorted(plan_scores, key=lambda a: a[1], reverse=True)
        best_50_plans = [p for p, s in plan_scores[:50]]

        return best_50_plans
