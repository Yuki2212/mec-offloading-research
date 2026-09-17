from __future__ import annotations

from typing import Any, Dict, Tuple, List
import random
import csv
import time
from pathlib import Path 
import numpy as np
from deap import tools

from mecoptimaloffloading.env_multi_mec import (
    SystemParams,
    Instance,
    generate_random_instance,
)
from mecoptimaloffloading.ga_multi_mec_energy import GAParams, make_toolbox
from mecoptimaloffloading.multicore_scheduler import (
    MultiCoreConfig,
    evaluate_with_multicore,
)

# Jain公平度(遅延) の下限値
FAIRNESS_MIN_DELAY = 0.85

# 公平性ペナルティ
FAIRNESS_PENALTY_WEIGHT = 10.0

# 起動MEC台数のペナルティ
ACTIVE_MEC_PENALTY_WEIGHT = 500.0 

# 起動台数突然変異率
SHUTDOWN_MUTATION_PROB = 0.35


def timed_run(fn, *args, **kwargs) -> tuple[Dict[str, Any], float]:
    """
    fnを実行
    """
    t0 = time.perf_counter()
    detail = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return detail, float(t1 - t0)


def total_energy(detail: Dict[str, Any]) -> float:
    """
    総消費エネルギー量:
      E_mec(=E_mec_task+E_mec_base) + E_cloud + sum(E_users)
    """
    return float(detail["E_mec"] + detail["E_cloud"] + np.sum(detail["E_users"]))


def jain_fairness(x: np.ndarray) -> float:
    """
    Jain's Fairness Index:
      F = ( (sum x_i)^2 ) / ( n * sum x_i^2 )
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return float("nan")
    if np.allclose(x, 0.0):
        return 1.0
    s1 = float(x.sum())
    s2 = float((x * x).sum())
    if s2 == 0.0:
        return float("nan")
    return (s1 * s1) / (n * s2)


def mec_counts_from_dst(dst: np.ndarray, N_MEC: int) -> np.ndarray:
    """MEC 0..N_MEC-1 の割当人数を返す（Cloud は除外）"""
    dst = np.asarray(dst, dtype=int)
    mec_only = dst[dst != N_MEC]
    if mec_only.size == 0:
        return np.zeros(N_MEC, dtype=int)
    return np.bincount(mec_only, minlength=N_MEC)


def print_summary(label: str, detail: Dict[str, Any], sys_params: SystemParams) -> None:
    """1 つの解のサマリ出力（共通フォーマット）"""
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    violations = detail["violations"]
    E_mec_total = float(detail["E_mec"])
    E_cloud = float(detail["E_cloud"])
    E_users = np.asarray(detail["E_users"], dtype=float)
    T_users = np.asarray(detail["T_users"], dtype=float)
    dst = np.asarray(detail["dst"], dtype=int)

    E_mec_task = float(detail.get("E_mec_task", E_mec_total))
    E_mec_base = float(detail.get("E_mec_base", 0.0))

    E_total = total_energy(detail)
    avg_delay = float(np.mean(T_users))
    avg_ue_energy = float(np.mean(E_users))

    num_viol = len(violations)
    viol_rate = 100.0 * num_viol / U

    num_mec_tasks = int((dst != N_MEC).sum())
    num_cloud_tasks = int((dst == N_MEC).sum())

    # 起動MEC台数
    num_active_mec = int(detail.get("num_active_mec", np.unique(dst[dst != N_MEC]).size))

    # Jain 公平度
    jain_delay = jain_fairness(T_users)
    jain_ue = jain_fairness(E_users)

    mec_counts = mec_counts_from_dst(dst, N_MEC)

    print(f"=== {label} ===")
    print(f"  violations = {violations}")
    print(
        f"  E_mec(total),E_cloud = {E_mec_total} {E_cloud}\n"
        f"    (E_mec_task={E_mec_task}, E_mec_base={E_mec_base})"
    )
    print(f"  総消費エネルギー量 = {E_total:.6f} J")
    print(f"  ユーザ平均遅延時間 = {avg_delay:.6f} s")
    print(f"  UE平均消費エネルギー = {avg_ue_energy:.6f} J")
    print(f"  締切違反タスク数 = {num_viol}")
    print(f"  締切違反率 = {viol_rate:.2f}%")
    print(f"  起動MEC数 = {num_active_mec}")
    print(f"  MECタスク数 = {num_mec_tasks} Cloudタスク数 = {num_cloud_tasks}")
    print(f"  mec_counts (per MEC 0..{N_MEC-1}) = {list(map(int, mec_counts))}")
    print(f"  Jain公平度(遅延) = {jain_delay:.6f}")
    print(f"  Jain公平度(UEエネルギー) = {jain_ue:.6f}")
    print(f"  dst = {list(map(int, dst))}\n")


# 目的関数

def objective_components(
    detail: Dict[str, Any],
    inst: Instance,
    sys_params: SystemParams,
) -> Dict[str, float]:
    T_users = np.asarray(detail["T_users"], dtype=float)

    base_energy = total_energy(detail)

    # ① 遅延違反ペナルティ
    delay_excess = np.maximum(T_users - inst.D_u, 0.0)
    sum_delay_excess = float(np.sum(delay_excess))
    delay_penalty = sys_params.penalty * sum_delay_excess / sys_params.num_users

    # ② Jain公平度(遅延) 下限ペナルティ
    jain_delay = jain_fairness(T_users)
    shortfall = max(0.0, FAIRNESS_MIN_DELAY - jain_delay)
    fairness_penalty = FAIRNESS_PENALTY_WEIGHT * shortfall

    # ③ 起動MEC台数ペナルティ（台数優先）
    num_active_mec = int(detail.get("num_active_mec", 0))
    active_mec_penalty = ACTIVE_MEC_PENALTY_WEIGHT * float(num_active_mec)

    return {
        "base_energy": float(base_energy),
        "sum_delay_excess": sum_delay_excess,
        "delay_penalty": float(delay_penalty),
        "jain_delay": float(jain_delay),
        "fairness_penalty": float(fairness_penalty),
        "num_active_mec": float(num_active_mec),
        "active_mec_penalty": float(active_mec_penalty),
    }


def objective_value(
    detail: Dict[str, Any],
    inst: Instance,
    sys_params: SystemParams,
) -> float:
    c = objective_components(detail, inst, sys_params)
    return (
        c["base_energy"]
        + c["delay_penalty"]
        + c["fairness_penalty"]
        + c["active_mec_penalty"]
    )


def print_breakdown(prefix: str, detail: Dict[str, Any], inst: Instance, sys_params: SystemParams) -> None:
    c = objective_components(detail, inst, sys_params)
    E_mec_task = float(detail.get("E_mec_task", np.nan))
    E_mec_base = float(detail.get("E_mec_base", np.nan))
    E_mec_total = float(detail.get("E_mec", np.nan))
    print(f"--- breakdown: {prefix} ---")
    print(f"  num_active_mec = {int(c['num_active_mec'])}")
    print(f"  E_mec_task = {E_mec_task:.6f}, E_mec_base = {E_mec_base:.6f}, E_mec_total = {E_mec_total:.6f}")
    print(f"  base_energy = {c['base_energy']:.6f}")
    print(f"  sum_delay_excess = {c['sum_delay_excess']:.6f}")
    print(f"  delay_penalty = {c['delay_penalty']:.6f}")
    print(f"  jain_delay = {c['jain_delay']:.6f}")
    print(f"  fairness_penalty = {c['fairness_penalty']:.6f}")
    print(f"  active_mec_penalty = {c['active_mec_penalty']:.6f}  (weight={ACTIVE_MEC_PENALTY_WEIGHT})")
    print(f"  fitness(total) = {objective_value(detail, inst, sys_params):.6f}\n")


# 1) GA

def _shutdown_mutation(individual, N_MEC: int, rng: random.Random) -> Tuple[bool, int]:
    """
    MEC台数突然変異
    """
    dst = list(map(int, individual))
    U = len(dst)
    cloud = N_MEC

    dst_arr = np.asarray(dst, dtype=int)
    active_mecs = np.unique(dst_arr[dst_arr != cloud])
    if active_mecs.size <= 1:
        return (False, -1)

    victim = int(rng.choice(active_mecs.tolist()))
    users_victim = np.where(dst_arr == victim)[0]
    if users_victim.size == 0:
        return (False, victim)

    survivors = [m for m in active_mecs.tolist() if m != victim]
    candidates = survivors + [cloud]

    changed = False
    for u in users_victim:
        new_dst = int(rng.choice(candidates))
        if new_dst != dst[u]:
            dst[u] = new_dst
            changed = True

    if changed:
        for i in range(U):
            individual[i] = int(dst[i])
    return (changed, victim)


def run_ga_active_mec_priority_multicore(
    label: str,
    inst: Instance,
    sys_params: SystemParams,
    ga_params: GAParams,
    mc_conf: MultiCoreConfig,
) -> Dict[str, Any]:
    """
    fitness = base_energy + delay_penalty + fairness_penalty + active_mec_penalty
    """
    sys_params.beta_task_e = ga_params.beta_task_e
    toolbox = make_toolbox(inst, sys_params, ga_params)

    rng = random.Random(ga_params.seed)

    def _eval(individual):
        dst = list(individual)
        detail = evaluate_with_multicore(dst, inst, sys_params, mc_conf)
        return (objective_value(detail, inst, sys_params),)

    toolbox.register("evaluate", _eval)

    original_mutate = toolbox.mutate

    def mixed_mutate(individual):
        if rng.random() < SHUTDOWN_MUTATION_PROB:
            _shutdown_mutation(individual, sys_params.num_mecs, rng)
            return (individual,)
        else:
            return original_mutate(individual)

    toolbox.register("mutate", mixed_mutate)

    pop = toolbox.population(n=ga_params.pop_size)

    # 初期評価
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    def detail_of(ind):
        return evaluate_with_multicore(list(ind), inst, sys_params, mc_conf)

    best_ind = tools.selBest(pop, 1)[0]
    best_detail = detail_of(best_ind)
    print(f"[GA init] base_energy={total_energy(best_detail):.6f}, fitness={objective_value(best_detail, inst, sys_params):.6f}, active_mec={best_detail.get('num_active_mec')}")
    print_breakdown("GA init(best)", best_detail, inst, sys_params)

    log_gens = {1, 5, 10, 20, 30, 40, 50, 75, 100, 150, ga_params.generations}

    for gen in range(1, ga_params.generations + 1):
        offspring = toolbox.select(pop, len(pop) - ga_params.elitism)
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.p_crossover:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        for ind in offspring:
            if random.random() < ga_params.p_mutation:
                toolbox.mutate(ind)
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values

        elites = tools.selBest(pop, ga_params.elitism)
        pop = offspring + elites

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        best_ind = tools.selBest(pop, 1)[0]
        if gen in log_gens:
            bd = detail_of(best_ind)
            print(f"[GA gen{gen}] base_energy={total_energy(bd):.6f}, fitness={objective_value(bd, inst, sys_params):.6f}, active_mec={bd.get('num_active_mec')}")
            print_breakdown(f"GA gen{gen}(best)", bd, inst, sys_params)

    final_best = tools.selBest(pop, 1)[0]
    final_detail = detail_of(final_best)

    violations = final_detail["violations"]
    feasible_deadline = (len(violations) == 0)

    print("\n[GA RESULT]")
    print("objective = base_energy + delay_penalty + fairness_penalty + active_mec_penalty")
    print(f"base_energy = {total_energy(final_detail):.6f}")
    print(f"fitness(total) = {objective_value(final_detail, inst, sys_params):.6f}\n")
    print_breakdown("GA final(best)", final_detail, inst, sys_params)

    summary = dict(final_detail)
    summary["label"] = label
    summary["feasible"] = feasible_deadline
    summary["score"] = float(total_energy(final_detail))  
    return summary


# 2) Greedy

def greedy_full_multicore_same_obj(
    inst: Instance,
    sys_params: SystemParams,
    mc_conf: MultiCoreConfig,
    max_passes: int = 1000,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    """
    ランダム初期解 → GA と同じ目的関数を局所探索で改善。
    """
    if rng is None:
        rng = np.random.default_rng()

    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    dst = rng.integers(0, N_MEC + 1, size=U, dtype=int)
    detail = evaluate_with_multicore(dst.tolist(), inst, sys_params, mc_conf)

    print("=== Greedy (random init) BEFORE local search ===")
    print_summary("Greedy BEFORE", detail, sys_params)
    print_breakdown("Greedy BEFORE", detail, inst, sys_params)

    all_choices = np.arange(N_MEC + 1, dtype=int)

    best_detail = detail
    best_obj = objective_value(detail, inst, sys_params)

    for _pass in range(max_passes):
        improved = False

        for u in range(U):
            current_choice = int(dst[u])

            local_best_choice = current_choice
            local_best_detail = best_detail
            local_best_obj = best_obj

            for choice in all_choices:
                if choice == current_choice:
                    continue

                dst[u] = int(choice)
                cand_detail = evaluate_with_multicore(dst.tolist(), inst, sys_params, mc_conf)
                cand_obj = objective_value(cand_detail, inst, sys_params)

                if cand_obj < local_best_obj:
                    local_best_obj = cand_obj
                    local_best_choice = int(choice)
                    local_best_detail = cand_detail

            dst[u] = local_best_choice
            if local_best_choice != current_choice:
                best_detail = local_best_detail
                best_obj = local_best_obj
                improved = True

        if not improved:
            break

    print_breakdown("Greedy final(best)", best_detail, inst, sys_params)
    return best_detail

# 3) CEGA
def run_ga_cega_like_multicore(
    label: str,
    inst: Instance,
    sys_params: SystemParams,
    ga_params: GAParams,
    mc_conf: MultiCoreConfig,
    w_delay: float = 1.0,
    w_ue: float = 1.0,
) -> Dict[str, Any]:
    """
    fitness:
      w_delay * 平均遅延 + w_ue * 平均UEエネルギ + penalty(締切超過)
    """
    sys_params.beta_task_e = ga_params.beta_task_e
    toolbox = make_toolbox(inst, sys_params, ga_params)

    def _eval(individual):
        dst = list(individual)
        detail = evaluate_with_multicore(dst, inst, sys_params, mc_conf)

        T_users = np.asarray(detail["T_users"], dtype=float)
        E_users = np.asarray(detail["E_users"], dtype=float)

        delay_excess = np.maximum(T_users - inst.D_u, 0.0)
        penalty = sys_params.penalty * float(np.sum(delay_excess)) / sys_params.num_users

        avg_delay = float(np.mean(T_users))
        avg_ue = float(np.mean(E_users))

        fitness = w_delay * avg_delay + w_ue * avg_ue + penalty
        return (fitness,)

    toolbox.register("evaluate", _eval)

    pop = toolbox.population(n=ga_params.pop_size)

    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    def detail_of(ind):
        return evaluate_with_multicore(list(ind), inst, sys_params, mc_conf)

    log_gens = {1, 5, 10, 20, 30, 40, 50, 75, 100, 150, ga_params.generations}

    for gen in range(1, ga_params.generations + 1):
        offspring = toolbox.select(pop, len(pop) - ga_params.elitism)
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.p_crossover:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        for ind in offspring:
            if random.random() < ga_params.p_mutation:
                toolbox.mutate(ind)
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values

        elites = tools.selBest(pop, ga_params.elitism)
        pop = offspring + elites

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        best_ind = tools.selBest(pop, 1)[0]
        if gen in log_gens:
            d = detail_of(best_ind)
            print(f"[CEGA gen{gen}] total_energy(ref)={total_energy(d):.6f}")

    best_ind = tools.selBest(pop, 1)[0]
    final_detail = detail_of(best_ind)
    final_detail["label"] = label
    final_detail["score"] = float(total_energy(final_detail))
    return final_detail

# 4) Random baseline

def random_assignment_multicore(
    inst: Instance,
    sys_params: SystemParams,
    mc_conf: MultiCoreConfig,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs
    dst = rng.integers(0, N_MEC + 1, size=U, dtype=int)
    detail = evaluate_with_multicore(dst.tolist(), inst, sys_params, mc_conf)
    detail["label"] = "Random_multicore"
    detail["score"] = float(total_energy(detail))
    return detail

# 統計

def collect_run_row(
    alg_key: str,
    detail: Dict[str, Any],
    inst: Instance,
    sys_params: SystemParams,
    scenario_idx: int,
    seed: int,
    runtime_sec: float, 
) -> Dict[str, Any]:
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    violations = detail["violations"]
    num_viol = int(len(violations))
    viol_rate = 100.0 * num_viol / U

    E_mec = float(detail["E_mec"])
    E_cloud = float(detail["E_cloud"])
    E_users = np.asarray(detail["E_users"], dtype=float)
    T_users = np.asarray(detail["T_users"], dtype=float)
    dst = np.asarray(detail["dst"], dtype=int)

    energy = total_energy(detail)
    avg_delay = float(np.mean(T_users))
    tail_delay_p95 = float(np.percentile(T_users, 95))
    max_delay = float(np.max(T_users))
    avg_ue_energy = float(np.mean(E_users))

    num_mec_tasks = int((dst != N_MEC).sum())
    num_cloud_tasks = int((dst == N_MEC).sum())
    num_active_mec = int(detail.get("num_active_mec", np.unique(dst[dst != N_MEC]).size))

    jain_delay = jain_fairness(T_users)
    jain_ue = jain_fairness(E_users)

    mec_counts = mec_counts_from_dst(dst, N_MEC)

    row: Dict[str, Any] = {
        "scenario": scenario_idx,
        "seed": seed,
        "algo": alg_key,
        "runtime_sec": float(runtime_sec), 
        "energy": energy,
        "violations": num_viol,
        "viol_rate": viol_rate,
        "avg_delay": avg_delay,
        "tail_delay_p95": tail_delay_p95,
        "max_delay": max_delay,
        "avg_ue_energy": avg_ue_energy,
        "E_mec": E_mec,
        "E_cloud": E_cloud,
        "num_mec_tasks": num_mec_tasks,
        "num_cloud_tasks": num_cloud_tasks,
        "num_active_mec": num_active_mec,  
        "jain_delay": jain_delay,
        "jain_ue_energy": jain_ue,
    }

    for i in range(N_MEC):
        row[f"mec_count_{i}"] = int(mec_counts[i])

    return row


def collect_per_user_rows(
    alg_key: str,
    detail: Dict[str, Any],
    inst: Instance,
    sys_params: SystemParams,
    scenario_idx: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    1行 = (scenario, algo, user) のデータを生成する。
    """
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs

    dst = np.asarray(detail["dst"], dtype=int)
    T_users = np.asarray(detail["T_users"], dtype=float)

    # シナリオ固定のユーザ情報
    D_u = np.asarray(inst.D_u, dtype=float)
    L_u = np.asarray(inst.L_u, dtype=float)
    O_u_bits = np.asarray(inst.O_u_bits, dtype=float)
    p_u = np.asarray(inst.p_u, dtype=float)
    gateway = np.asarray(inst.gateway, dtype=int)

    excess = np.maximum(T_users - D_u, 0.0)
    slack = D_u - T_users


    assigned_server = np.asarray(detail.get("assigned_server", np.full(U, -1)), dtype=int)
    assigned_cores = detail.get("assigned_cores", [[] for _ in range(U)])
    assigned_cores_len = np.asarray([len(x) for x in assigned_cores], dtype=int)

    rows: List[Dict[str, Any]] = []
    for u in range(U):
        rows.append({
            "scenario": scenario_idx,
            "seed": seed,
            "algo": alg_key,
            "user": u,
            "dst": int(dst[u]),
            "is_cloud": int(dst[u] == N_MEC),

            "deadline": float(D_u[u]),
            "delay": float(T_users[u]),
            "excess": float(excess[u]),  
            "slack": float(slack[u]),    

            "L_u": float(L_u[u]),
            "O_u_bits": float(O_u_bits[u]),
            "p_u": float(p_u[u]),
            "gateway": int(gateway[u]),

            "assigned_server": int(assigned_server[u]),
            "assigned_cores_len": int(assigned_cores_len[u]),
        })
    return rows


def summarize_and_rank(rows_run: list[Dict[str, Any]], alg_names: list[str], metric_names: list[str]) -> None:
    print()
    print("=== Summary over all scenarios (mean only) ===")
    print("====================================================\n")

    means: Dict[str, Dict[str, float]] = {alg: {} for alg in alg_names}

    for alg in alg_names:
        print(f"--- {alg} ---")
        alg_rows = [r for r in rows_run if r["algo"] == alg]
        for metric in metric_names:
            arr = np.asarray([float(r[metric]) for r in alg_rows], dtype=float)
            mean_val = float(arr.mean()) if arr.size > 0 else float("nan")
            means[alg][metric] = mean_val
            print(f"{metric:>18}: mean = {mean_val:.6f}")
        print()

    def rank(metric_key: str, higher_is_better: bool = False) -> list[tuple[str, float]]:
        items = [(alg, means[alg][metric_key]) for alg in alg_names]
        items.sort(key=lambda x: x[1], reverse=higher_is_better)
        return items

    def print_ranking(metric_key: str, title: str, unit: str = "", higher_is_better: bool = False) -> None:
        print(f"\n=== Ranking: {title} (mean, {'higher' if higher_is_better else 'lower'} is better) ===")
        for i, (alg, val) in enumerate(rank(metric_key, higher_is_better), start=1):
            if unit:
                print(f"{i}. {alg}: {val:.6f} {unit}")
            else:
                print(f"{i}. {alg}: {val:.6f}")


    print_ranking("runtime_sec", "runtime", "sec", higher_is_better=False)  
    print_ranking("energy", "total energy", "J", higher_is_better=False)
    print_ranking("violations", "deadline violations", higher_is_better=False)
    print_ranking("avg_delay", "avg delay", "s", higher_is_better=False)
    print_ranking("tail_delay_p95", "tail delay p95", "s", higher_is_better=False)
    print_ranking("max_delay", "max delay", "s", higher_is_better=False)
    print_ranking("avg_ue_energy", "avg UE energy", "J", higher_is_better=False)
    print_ranking("E_mec", "E_mec", "J", higher_is_better=False)
    print_ranking("E_cloud", "E_cloud", "J", higher_is_better=False)
    print_ranking("num_active_mec", "num active MEC", "servers", higher_is_better=False)
    print_ranking("jain_delay", "Jain fairness (delay)", higher_is_better=True)
    print_ranking("jain_ue_energy", "Jain fairness (UE energy)", higher_is_better=True)


# main

def main(config: dict, output_dir: Path) -> None:
    sys_params = SystemParams(**config["system"])
    mc_conf = MultiCoreConfig(
        mec_cores=np.array([config["multicore"]["cores_per_mec"]] * sys_params.num_mecs, dtype=int),
        alpha_parallel=config["multicore"]["alpha_parallel"],
    )
    ga_params = GAParams(**config["ga"])
    num_scenarios = config["num_scenarios"]

    alg_names = ["GA", "Greedy", "CEGA", "Random"]

    metric_names = [
        "runtime_sec", 
        "energy",
        "violations",
        "viol_rate",
        "avg_delay",
        "tail_delay_p95",
        "max_delay",
        "avg_ue_energy",
        "E_mec",
        "E_cloud",
        "num_mec_tasks",
        "num_cloud_tasks",
        "num_active_mec",
        "jain_delay",
        "jain_ue_energy",
    ]

    rows_run: list[Dict[str, Any]] = []
    rows_user_long: list[Dict[str, Any]] = []  

    for scenario_idx in range(1, num_scenarios + 1):
        seed = scenario_idx

        print()
        print("#" * 52)
        print(f"### Scenario {scenario_idx} / {num_scenarios} (seed={seed})")
        print("#" * 52)
        print()

        inst = generate_random_instance(sys_params, seed=seed)

        # GA
        print("=== GA (ACTIVE MEC PRIORITY, multicore) ===")
        ga_detail, ga_time = timed_run(
            run_ga_active_mec_priority_multicore,
            "GA_active_mec_priority_multicore", inst, sys_params, ga_params, mc_conf
        )
        print(f"[Timing] GA runtime = {ga_time:.6f} sec")
        print_summary("GA_active_mec_priority_multicore", ga_detail, sys_params)
        rows_run.append(collect_run_row("GA", ga_detail, inst, sys_params, scenario_idx, seed, ga_time))
        rows_user_long.extend(collect_per_user_rows("GA", ga_detail, inst, sys_params, scenario_idx, seed))

        # Greedy 
        print("=== Greedy (same objective as GA, multicore) ===")
        greedy_detail, greedy_time = timed_run(
            greedy_full_multicore_same_obj,
            inst, sys_params, mc_conf,
            max_passes=1000
        )
        print(f"[Timing] Greedy runtime = {greedy_time:.6f} sec")
        greedy_detail["label"] = "Greedy_full_multicore_same_obj"
        greedy_detail["score"] = float(total_energy(greedy_detail))
        print_summary("Greedy_full_multicore_same_obj", greedy_detail, sys_params)
        rows_run.append(collect_run_row("Greedy", greedy_detail, inst, sys_params, scenario_idx, seed, greedy_time))
        rows_user_long.extend(collect_per_user_rows("Greedy", greedy_detail, inst, sys_params, scenario_idx, seed))

        #　CEGA
        print("=== CEGA-like GA (delay + UE energy, multicore) ===")
        cega_detail, cega_time = timed_run(
            run_ga_cega_like_multicore,
            "GA_CEGA_like_multicore", inst, sys_params, ga_params, mc_conf,
            w_delay=1.0, w_ue=1.0
        )
        print(f"[Timing] CEGA runtime = {cega_time:.6f} sec")
        print_summary("GA_CEGA_like_multicore", cega_detail, sys_params)
        rows_run.append(collect_run_row("CEGA", cega_detail, inst, sys_params, scenario_idx, seed, cega_time))
        rows_user_long.extend(collect_per_user_rows("CEGA", cega_detail, inst, sys_params, scenario_idx, seed))

        # Random
        print("=== Random assignment baseline (multicore) ===")
        rand_detail, rand_time = timed_run(random_assignment_multicore, inst, sys_params, mc_conf)
        print(f"[Timing] Random runtime = {rand_time:.6f} sec")
        print_summary("Random_multicore", rand_detail, sys_params)
        rows_run.append(collect_run_row("Random", rand_detail, inst, sys_params, scenario_idx, seed, rand_time))
        rows_user_long.extend(collect_per_user_rows("Random", rand_detail, inst, sys_params, scenario_idx, seed))

    summarize_and_rank(rows_run, alg_names, metric_names)

    run_fieldnames = [
        "scenario",
        "seed",
        "algo",
        "runtime_sec",  
        "energy",
        "violations",
        "viol_rate",
        "avg_delay",
        "tail_delay_p95",
        "max_delay",
        "avg_ue_energy",
        "E_mec",
        "E_cloud",
        "num_mec_tasks",
        "num_cloud_tasks",
        "num_active_mec",  
        "jain_delay",
        "jain_ue_energy",
    ] + [f"mec_count_{i}" for i in range(sys_params.num_mecs)]

    out_csv = output_dir / f"results_compare_all_no_capacity{config['condition_id']}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fieldnames)
        writer.writeheader()
        writer.writerows(rows_run)

    print(f"\nSaved CSV file: {out_csv}")

    per_user_fieldnames = [
        "scenario",
        "seed",
        "algo",
        "user",
        "dst",
        "is_cloud",
        "deadline",
        "delay",
        "excess",  
        "slack", 
        "L_u",
        "O_u_bits",
        "p_u",
        "gateway",
        "assigned_server",
        "assigned_cores_len",
    ]

    out_user_csv = output_dir / f"results_per_user_long_no_capacity{config['condition_id']}.csv"
    with open(out_user_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_user_fieldnames)
        writer.writeheader()
        writer.writerows(rows_user_long)

    print(f"Saved per-user long CSV file: {out_user_csv}")


