from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

import random
import numpy as np
from deap import base, creator, tools

from .env_multi_mec import SystemParams, Instance

# GAパラメータ

@dataclass
class GAParams:
    pop_size: int
    generations: int
    p_crossover: float
    p_mutation: float
    tournament_k: int
    elitism: int
    seed: int
    beta_task_e: float = 0.5   
    balance: float = 0.0   

# 評価関数
    
def _evaluate_core(
    individual: List[int],
    inst: Instance,
    sys_params: SystemParams,
) -> Dict[str, Any]:
    """
    与えられた割当に対して、エネルギー・遅延・締切違反を計算
    """
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    dst = np.asarray(individual, dtype=int)
    assert dst.shape == (U,)

    idx_mec: List[List[int]] = [[] for _ in range(N_MEC)]
    idx_cloud: List[int] = []

    for u, choice in enumerate(dst):
        if choice == N_MEC:
            idx_cloud.append(u)
        else:
            if 0 <= choice < N_MEC:
                idx_mec[choice].append(u)
            else:
                idx_cloud.append(u)


    T_users = np.zeros(U)     # 総遅延 [s]
    E_users = np.zeros(U)     # UE側エネルギ [J]
    E_mec = 0.0
    E_cloud = 0.0


    for m in range(N_MEC):
        users_m = idx_mec[m]
        if not users_m:
            continue

        users_m = np.asarray(users_m, dtype=int)
        n_m = len(users_m)

        f_each = sys_params.f_mec / n_m

        h = inst.h_um[users_m, m]
        R_ul = sys_params.W * np.log2(1.0 + inst.p_u[users_m] * h / sys_params.sigma)
        t_ul = inst.O_u_bits[users_m] / R_ul  

        t_exec = inst.L_u[users_m] / f_each  

        T_users[users_m] = t_ul + t_exec
        E_users[users_m] = inst.p_u[users_m] * t_ul

        E_mec += sys_params.e_cycle_mec * float(np.sum(inst.L_u[users_m]))

    if idx_cloud:
        users_c = np.asarray(idx_cloud, dtype=int)
        n_c = len(users_c)

        f_each_cloud = sys_params.f_cloud / n_c

        gw = inst.gateway[users_c]  
        h_gw = inst.h_um[users_c, gw]
        R_ul_c = sys_params.W * np.log2(1.0 + inst.p_u[users_c] * h_gw / sys_params.sigma)
        t_ul_c = inst.O_u_bits[users_c] / R_ul_c

        # バックホール遅延
        t_bh_up = inst.O_u_bits[users_c] / sys_params.R_BH_up
        t_bh_down = inst.O_u_bits[users_c] / sys_params.R_BH_down

        # クラウド計算遅延
        t_exec_c = inst.L_u[users_c] / f_each_cloud

        # 総遅延， UEエネルギー
        T_users[users_c] = (
            t_ul_c + t_bh_up + t_exec_c + t_bh_down + sys_params.ell_cloud
        )
        E_users[users_c] = inst.p_u[users_c] * t_ul_c

        # クラウド＋BHエネルギー
        E_cloud += sys_params.e_cycle_cloud * float(np.sum(inst.L_u[users_c]))
        E_cloud += sys_params.e_BH_up * float(np.sum(inst.O_u_bits[users_c]))
        E_cloud += sys_params.e_BH_down * float(np.sum(inst.O_u_bits[users_c]))

    E_users_sum = float(np.sum(E_users))
    E_comp = E_mec + E_cloud
    base_energy = E_comp + E_users_sum

    delay_excess = np.maximum(T_users - inst.D_u, 0.0)
    violations_idx = np.nonzero(delay_excess > 0.0)[0].tolist()

    penalty_cost = (
        sys_params.penalty * float(np.sum(delay_excess)) / sys_params.num_users
    )

    fitness = base_energy + penalty_cost

    return {
        "fitness": fitness,
        "score": base_energy,   
        "E_mec": E_mec,
        "E_cloud": E_cloud,
        "E_users": E_users,
        "T_users": T_users,
        "dst": dst,
        "violations": violations_idx,
    }

def _make_toolbox(
    inst: Instance,
    sys_params: SystemParams,
    ga_params: GAParams,
) -> base.Toolbox:
    """DEAP の toolbox を構築"""
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    random.seed(ga_params.seed)
    np.random.seed(ga_params.seed)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 目的変数
    toolbox.register("attr_dst", random.randint, 0, N_MEC)

    # 個体
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_dst,
        n=U,
    )

    # 集団
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 評価関数
    def _eval(individual):
        detail = _evaluate_core(individual, inst, sys_params)
        return (detail["fitness"],)

    toolbox.register("evaluate", _eval)

    # 選択・交叉・突然変異
    toolbox.register(
        "select", tools.selTournament, tournsize=ga_params.tournament_k
    )
    toolbox.register("mate", tools.cxTwoPoint)

    def _mutate(individual):
        # 各遺伝子を小さな確率で別のサーバに振り直す
        for i in range(len(individual)):
            if random.random() < 1.0 / U:
                individual[i] = random.randint(0, N_MEC)
        return (individual,)

    toolbox.register("mutate", _mutate)

    return toolbox

make_toolbox = _make_toolbox

# GA 本体

def run_ga(
    label: str,
    inst: Instance,
    sys_params: SystemParams,
    ga_params: GAParams,
) -> Dict[str, Any]:
    """
    GA を実行し、ログ出力と結果まとめを返す。
    label: "A_base" などの名前
    """
    toolbox = _make_toolbox(inst, sys_params, ga_params)

    print(f"=== GA {label} (objective='energy') ===")
    print("  [GAパラメータ]")
    print(f"    個体数(pop_size)        = {ga_params.pop_size}")
    print(f"    世代数(generations)     = {ga_params.generations}")
    print(f"    交叉確率(p_crossover)   = {ga_params.p_crossover}")
    print(f"    突然変異率(p_mutation)  = {ga_params.p_mutation}")
    print(f"    トーナメントサイズ(k)   = {ga_params.tournament_k}")
    print(f"    エリート数(elitism)     = {ga_params.elitism}")
    print(f"    乱数シード(seed)        = {ga_params.seed}")
    print(f"    beta_task_e             = {ga_params.beta_task_e}")
    print(f"    balance                 = {ga_params.balance}")

    sys_params.beta_task_e = ga_params.beta_task_e

    # 初期集団
    pop = toolbox.population(n=ga_params.pop_size)

    # 初期個体の評価
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    # 初期ベスト
    best_ind = tools.selBest(pop, 1)[0]
    init_detail = _evaluate_core(best_ind, inst, sys_params)

    # ログ出力する世代
    log_gens = {
        1, 5, 10, 15, 20, 25, 30, 35,
        40, 45, 50, 55, 60, 65, 70,
        75, 80, 85, ga_params.generations,
    }

    # 世代ループ
    for gen in range(1, ga_params.generations + 1):
        # 次世代個体を選択
        offspring = toolbox.select(pop, len(pop) - ga_params.elitism)
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.p_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 突然変異
        for ind in offspring:
            if random.random() < ga_params.p_mutation:
                toolbox.mutate(ind)
                del ind.fitness.values

        # エリート保存
        elites = tools.selBest(pop, ga_params.elitism)
        pop = offspring + elites

        # 評価が必要な個体
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # 世代ベスト
        best_ind = tools.selBest(pop, 1)[0]

        if gen in log_gens:
            detail = _evaluate_core(best_ind, inst, sys_params)
         

    # 最終結果
    final_detail = _evaluate_core(best_ind, inst, sys_params)

    violations = final_detail["violations"]
    feasible = (len(violations) == 0)

    print("\n[GA RESULT]")
    print(f"feasible = {feasible}")
    if violations:
        msgs = []
        for u in violations:
            T = final_detail["T_users"][u]
            D = inst.D_u[u]
            msgs.append(f"user{u+1}: T={T:.3g}s > D={D:.3g}s")
        print(f"violations = {msgs}")
    else:
        print("violations = []")
    print("objective = energy")
    print(f"score = {final_detail['score']}")
    print(f"E_mec,E_cloud = {final_detail['E_mec']} {final_detail['E_cloud']}")
    print(f"E_users = {list(map(float, final_detail['E_users']))}")
    print(f"T_users = {list(map(float, final_detail['T_users']))}")
    print(f"dst = {list(map(int, final_detail['dst']))}\n")

    summary = {
        "label": label,
        "feasible": feasible,
        "violations": violations,
        "score": final_detail["score"],
        "E_mec": final_detail["E_mec"],
        "E_cloud": final_detail["E_cloud"],
        "E_users": final_detail["E_users"],
        "T_users": final_detail["T_users"],
        "dst": final_detail["dst"],
    }

    return summary