# https://projecteuler.net/problem=240
"""
There are 1111 ways in which five 6-sided dice (sides numbered 1 to 6) can be rolled so that the top three sum to 15.
Some examples are:

D1,D2,D3,D4,D5 = 4,3,6,3,5
D1,D2,D3,D4,D5 = 4,3,3,5,6
D1,D2,D3,D4,D5 = 3,3,3,6,6
D1,D2,D3,D4,D5 = 6,6,3,3,3

In how many ways can twenty 12-sided dice (sides numbered 1 to 12) be rolled so that the top ten sum to 70?
"""

import math
import enum
from dataclasses import dataclass
from typing import List, Optional, Dict
import itertools


@dataclass
class TopDiceProblemInfo:
    num_dice_sides: int
    num_throws: int
    num_top_throws: int
    target_sum: int


@dataclass
class CombinationProblemInfo:
    max_in_comb: int
    min_in_comb: int
    num_items: int
    target_sum: int


def yield_combinations(info: CombinationProblemInfo, current_comb: List[int], current_max: int) -> Optional[List[int]]:
    plog(
        f"yield_combinations: Enter. list_of_top_throws: {current_comb}. current_max_top_throw={current_max}", LogLevel.debug
    )
    num_of_next_items = info.num_items - len(current_comb)
    assert num_of_next_items >= 0
    current_sum = sum(current_comb)
    if current_sum > info.target_sum:  # no negative items in the combination
        plog(
            f"yield_combinations: invalid. list: {current_comb}",
            LogLevel.debug,
        )
        return None
    elif num_of_next_items == 0:
        if current_sum == info.target_sum:
            plog(f"yield_combinations: solution found. list: {current_comb}", LogLevel.debug)
            return current_comb
        else:
            plog(
                f"yield_combinations: invalid. list: {current_comb}, sum: {current_sum}",
                LogLevel.debug,
            )
            return None
    else:
        m = current_max
        needed_sum = info.target_sum - current_sum
        while m >= info.min_in_comb and m * num_of_next_items >= needed_sum:
            plog(
                f"yield_combinations: enter-while. "
                f"m: {m}, "
                f"num_of_next_items: {num_of_next_items}, "
                f"needed_sum: {needed_sum}",
                LogLevel.debug,
            )
            assert m >= info.min_in_comb, f'm: {m}, min_in_comb: {info.min_in_comb}'
            next_list = current_comb[:]
            next_list.append(m)
            plog(f"yield_combinations: After adding. list: {next_list}", LogLevel.debug)
            next_comb = yield from yield_combinations(info, current_comb=next_list, current_max=m)
            plog(
                f"yield_combinations: after yield from. "
                f"m: {m}, "
                f"next_list: {next_list}, "
                f"next_comb: {next_comb}, ",
                LogLevel.debug
            )
            if next_comb:
                yield next_comb
            else:
                plog(
                    f"yield_combinations: next_comb is None. "
                    f"m: {m}, "
                    f"next_list: {next_list}, "
                    f"next_comb: {next_comb}, ",
                    LogLevel.debug
                )
            m = min(m-1, needed_sum)


def extract_occurrences(vector: List[int]) -> Dict[int, int]:
    occ = {}
    for n in vector:
        occ[n] = occ.get(n, 0) + 1
    return occ


def get_non_comb_occurrences(problem: TopDiceProblemInfo, comb: List[int],
                             reset_cache=True, cache={}) -> List[Dict[int, int]]:
    plog(
        f"get_non_comb_occurrences: enter. "
        f"problem: {problem}, "
        f"comb: {comb}, "
        f"reset_cache: {reset_cache}, "
        f"cache: {cache}",
        LogLevel.debug
    )
    max_non_comb_occurrences = problem.num_throws - problem.num_top_throws
    new_problem = CombinationProblemInfo(max_in_comb=max_non_comb_occurrences, min_in_comb=0,
                                         num_items=min(comb), target_sum=problem.num_throws-problem.num_top_throws)
    if reset_cache:
        cache = {}
    cache_hash = new_problem.num_items
    plog(
        f"get_non_comb_occurrences. "
        f"new_problem: {new_problem}",
        LogLevel.debug
    )
    if new_problem.num_items not in cache:
        cache[cache_hash] = []
        combs = yield_combinations(info=new_problem, current_comb=[], current_max=new_problem.max_in_comb)
        for comb in combs:
            plog(
                f"get_non_comb_occurrences: for comb in combs. "
                f"comb: {comb}",
                LogLevel.debug
            )
            s = set()
            comb_perms = itertools.permutations(comb, len(comb))
            for c_p in comb_perms:
                plog(
                    f"get_non_comb_occurrences: for c_p in comb_perms. "
                    f"c_p: {c_p}",
                    LogLevel.debug
                )
                if c_p not in s:
                    s.add(c_p)
                    d = {k: v for (k, v) in [(i + 1, c_p[i]) for i in range(new_problem.num_items)]}
                    plog(
                        f"get_non_comb_occurrences: c_p not in s. "
                        f"c_p: {c_p}, "
                        f"d: {d}",
                        LogLevel.debug
                    )
                    cache[cache_hash].append(d)
                else:
                    plog(
                        f"get_non_comb_occurrences: c_p in s already "
                        f"c_p: {c_p}, ",
                        LogLevel.debug
                    )
    return cache[cache_hash]


def merge_dicts(d1: Dict[int, int], d2: Dict[int, int]) -> Dict[int, int]:
    merged = {}
    for k, v in d1.items():
        merged[k] = v
    for k, v in d2.items():
        merged[k] = merged.get(k, 0) + d2[k]
    return merged


def calc_num_throws_per_comb(problem: TopDiceProblemInfo, comb: List[int]) -> int:
    plog(
        f"calc_num_throws_per_comb: enter. "
        f"comb: {comb}, ",
        LogLevel.debug
    )
    num_throws = 0
    nominator = math.factorial(problem.num_throws)
    comb_occurrences = extract_occurrences(comb)
    plog(
        f"calc_num_throws_per_comb: general computations. "
        f"comb: {comb}, "
        f"comb_occurrences: {comb_occurrences}, "
        f"nominator: {nominator}, ",
        LogLevel.debug
    )
    for non_comb_occ in get_non_comb_occurrences(problem, comb, reset_cache=True):
        plog(
            f"calc_num_throws_per_comb: got non_comb_occ. "
            f"non_comb_occ: {non_comb_occ}, ",
            LogLevel.debug
        )
        all_occurrences = merge_dicts(comb_occurrences, non_comb_occ)
        plog(
            f"calc_num_throws_per_comb: all_occurrences of comb. "
            f"comb: {comb}, "
            f"non_comb_occ: {non_comb_occ}, "
            f"all_occurrences: {all_occurrences}",
            LogLevel.debug
        )
        denom = 1
        for n in all_occurrences.values():
            denom *= math.factorial(n)
        num_throws += nominator/denom
        plog(
            f"calc_num_throws_per_comb: all_occurrences of comb. "
            f"comb: {comb}, "
            f"non_comb_occ: {non_comb_occ}, "
            f"all_occurrences: {all_occurrences}, "
            f"denom: {denom}, "
            f"num_throws: {num_throws}",
            LogLevel.debug
        )
    return num_throws


def solve(problem: TopDiceProblemInfo) -> int:
    sub_solution_by_comb = {}
    info = CombinationProblemInfo(max_in_comb=problem.num_dice_sides, min_in_comb=1, num_items=problem.num_top_throws,
                                  target_sum=problem.target_sum)
    comb_num = 0
    try:
        for comb in yield_combinations(info, current_comb=[], current_max=problem.num_dice_sides):
            comb_hash = str.join(",", [str(c) for c in comb])
            if DEBUG_MODE:
                assert comb_hash not in sub_solution_by_comb
            comb_num += 1
            num_throws_per_comb = calc_num_throws_per_comb(problem, comb)
            plog(
                f"solve_2nd_way: got comb. "
                f"comb_num: {comb_num}, "
                f"comb: {comb}, "
                f"num_throws_per_comb: {num_throws_per_comb}",
                LogLevel.info
            )
            sub_solution_by_comb[comb_hash] = num_throws_per_comb
    except StopIteration:
        plog(
            f"solve_2nd_way: Iteration stopped. ", LogLevel.debug
        )
    solution = 0
    plog(
        f"solve_2nd_way: summarizing dict"
        f"sub_solution_by_comb: {sub_solution_by_comb}, ",
        LogLevel.debug
    )
    for sub in sub_solution_by_comb.values():
        solution += sub
    return solution


"""
General section 
"""

DEBUG_MODE = True


class LogLevel(enum.IntEnum):
    none = 0
    info = 1
    debug = 2


LL = LogLevel.info


def plog(msg, ll=LogLevel.info):
    if ll <= LL:
        print(msg)


PROBLEM_ONE_DICT = {
    "num_dice_sides": 6,
    "num_throws": 5,
    "num_top_throws": 3,
    "target_sum": 15
}

PROBLEM_ONE = TopDiceProblemInfo(num_dice_sides=6,
                                 num_throws=5,
                                 num_top_throws=3,
                                 target_sum=15)

PROBLEM_TWO = TopDiceProblemInfo(num_dice_sides=12,
                                 num_throws=20,
                                 num_top_throws=10,
                                 target_sum=70)


def main():
    number = solve(PROBLEM_TWO)
    print(f"solution is: {int(number)}")


if __name__ == "__main__":
    main()
