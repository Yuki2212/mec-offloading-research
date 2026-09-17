from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mecoptimaloffloading.env_multi_mec import SystemParams, Instance
from mecoptimaloffloading.ga_multi_mec_energy import _evaluate_core


@dataclass
class MultiCoreConfig:
    """
    MECごとのマルチコア構成と並列化パラメータ
    """
    mec_cores: np.ndarray
    alpha_parallel: float = 0.9


def _exec_time_mec_multicore(
    L_u: float,
    f_mec: float,
    m_i: int,
    alpha_parallel: float,
) -> float:
    """
    マルチコア実行時間 t_e,ij を近似的に計算
    """
    m_i = max(1, m_i)
    s_i = 1.0 / (1.0 - alpha_parallel + alpha_parallel / float(m_i))
    return L_u / (s_i * f_mec)


def schedule_mec_multicore(
    dst: np.ndarray,
    inst: Instance,
    sys_params: SystemParams,
    mc_conf: MultiCoreConfig,
    m_per_task: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    スケジューラ
    """
    U = sys_params.num_users
    N_MEC = sys_params.num_mecs
    dst_arr = np.asarray(dst, dtype=int)

    if mc_conf.mec_cores.shape[0] != N_MEC:
        raise ValueError(
            f"len(mec_cores)={mc_conf.mec_cores.shape[0]} と "
            f"num_mecs={N_MEC} が一致していません"
        )

    if m_per_task is None:
        m_per_task = np.ones(U, dtype=int)
    else:
        m_per_task = np.asarray(m_per_task, dtype=int)
        if m_per_task.shape[0] != U:
            raise ValueError("m_per_task の長さが num_users と一致していません")

    max_M = int(np.max(mc_conf.mec_cores))
    ct = np.zeros((N_MEC, max_M), dtype=float)

    T_exec_mec = np.zeros(U, dtype=float)


    assigned_server = np.full(U, -1, dtype=int)
    assigned_cores: list[list[int]] = [[] for _ in range(U)]

    f_mec = sys_params.f_mec 

    for i_mec in range(N_MEC):
        users_i = np.where(dst_arr == i_mec)[0]
        if users_i.size == 0:
            continue

        M_i = int(mc_conf.mec_cores[i_mec])
        if M_i <= 0:
            continue

        ct_i = ct[i_mec, :M_i]


        for u in users_i:
            m_i = int(m_per_task[u])
            if m_i < 1:
                m_i = 1
            if m_i > M_i:
                m_i = M_i

            core_idx_sorted = np.argsort(ct_i)
            chosen = core_idx_sorted[:m_i]

            start_time = float(np.max(ct_i[chosen]))

            L_u = float(inst.L_u[u])
            t_exec = _exec_time_mec_multicore(
                L_u=L_u,
                f_mec=f_mec,
                m_i=m_i,
                alpha_parallel=mc_conf.alpha_parallel,
            )

            finish_time = start_time + t_exec

            ct_i[chosen] = finish_time

            assigned_server[u] = i_mec
            assigned_cores[u] = chosen.tolist()
            T_exec_mec[u] = finish_time

        ct[i_mec, :M_i] = ct_i

    return {
        "T_exec_mec": T_exec_mec,
        "ct_table": ct,
        "assigned_server": assigned_server,
        "assigned_cores": assigned_cores,
    }


def evaluate_with_multicore(
    dst: list[int],
    inst: Instance,
    sys_params: SystemParams,
    mc_conf: MultiCoreConfig,
) -> Dict[str, Any]:
    """
    評価関数
    """
    base = _evaluate_core(dst, inst, sys_params)

    U = sys_params.num_users
    N_MEC = sys_params.num_mecs
    dst_arr = np.asarray(dst, dtype=int)

    mc_res = schedule_mec_multicore(
        dst=dst_arr,
        inst=inst,
        sys_params=sys_params,
        mc_conf=mc_conf,
    )
    T_exec_mec = mc_res["T_exec_mec"]

    T_users = np.array(base["T_users"], copy=True)

    f_mec = sys_params.f_mec
    for m in range(N_MEC):
        users_i = np.where(dst_arr == m)[0]
        if users_i.size == 0:
            continue
        old_exec = inst.L_u[users_i] / f_mec
        T_users[users_i] -= old_exec
        T_users[users_i] += T_exec_mec[users_i]

    base["T_users"] = T_users

    base["assigned_server"] = mc_res["assigned_server"]
    base["assigned_cores"] = mc_res["assigned_cores"]
    base["ct_table"] = mc_res["ct_table"]

    mec_only = dst_arr[dst_arr != N_MEC]
    if mec_only.size == 0:
        num_active_mec = 0
    else:
        num_active_mec = int(np.unique(mec_only).size)

    E_mec_task = float(base["E_mec"])
    e_mec_boot = float(getattr(sys_params, "e_mec_boot", 0.0))
    E_mec_base = e_mec_boot * num_active_mec

    base["E_mec_task"] = E_mec_task
    base["E_mec_base"] = E_mec_base
    base["E_mec"] = E_mec_task + E_mec_base
    base["num_active_mec"] = num_active_mec

    return base
