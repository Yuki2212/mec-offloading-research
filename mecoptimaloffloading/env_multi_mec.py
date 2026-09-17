
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SystemParams:
    num_users: int = 100           # ユーザ数(タスク数）
    num_mecs: int = 8              # MECサーバ台数


    W: float = 80e6                # 帯域 [Hz]
    sigma: float = 1.0e-9          # 受信雑音電力 [W]
    P_min: float = 0.10            # UE送信電力の下限 [W]
    P_max: float = 0.20            # UE送信電力の上限 [W]
    d_min: float = 20.0            # UE-MEC距離の下限 [m]
    d_max: float = 80.0            # UE-MEC距離の上限 [m]
    c: float = 3.0e8               # 光速 [m/s]
    fc: float = 915e6              # キャリア周波数 [Hz]

    L_candidates: Tuple[float, ...] = (1e9, 1.5e9, 2e9)
    O_candidates_kB: Tuple[int, ...] = (100, 150, 200)
    D_min: float = 0.10            # 締切下限 [s]
    D_max: float = 0.12            # 締切上限 [s]

    f_mec: float = 5.0e11          # MEC 1コアあたりのクロック [cycle/s]
    f_cloud: float = 1.5e12        # クラウドのクロック [cycle/s]
    e_cycle_mec: float = 3.0e-12   # MEC 1サイクルあたりエネルギ [J]
    e_cycle_cloud: float = 1.5e-12 # クラウド 1サイクルあたりエネルギ [J]


    e_mec_boot: float = 1.0        # MEC起動電力[J] 


    R_bh_up: float = 1.0e10        # ULバックホールレート [bit/s]
    R_bh_down: float = 1.0e10      # DLバックホールレート [bit/s]
    e_bh_up: float = 1.0e-9        # ULバックホールエネルギ [J/bit]
    e_bh_down: float = 1.0e-9      # DLバックホールエネルギ [J/bit]
    ell_cloud: float = 0.01        # クラウド処理固定遅延 [s]


    k_local: float = 1e-27         # UEローカル計算のエネルギ係数
    beta_task_e: float = 0.5       # GAのエネルギ重み（外から上書きすることもある）
    penalty: float = 2.0e5         # 締切違反ペナルティ係数


    @property
    def U(self) -> int:
        return self.num_users

    @property
    def N_MEC(self) -> int:
        return self.num_mecs

    @property
    def f_MEC(self) -> float:
        return self.f_mec

    @property
    def e_cycle_MEC(self) -> float:
        return self.e_cycle_mec

    @property
    def R_BH_up(self) -> float:
        return self.R_bh_up

    @property
    def R_BH_down(self) -> float:
        return self.R_bh_down

    @property
    def e_BH_up(self) -> float:
        return self.e_bh_up

    @property
    def e_BH_down(self) -> float:
        return self.e_bh_down


@dataclass
class Instance:
    L_u: np.ndarray          # 計算量[cycle]
    O_u_bits: np.ndarray     # アップロードサイズ[bit]
    D_u: np.ndarray          # 締切[s]

    p_u: np.ndarray          # 送信電力[W]

    d_um: np.ndarray         # UE-MEC距離[m]
    h_um: np.ndarray         # ULチャネル利得
    g_um: np.ndarray         # チャネル利得
    gateway: np.ndarray      # クラウドに送るときの経由MEC ID


def _gen_task_params(params: SystemParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """タスクの L_u, O_u_bits, D_u を生成"""
    U = params.num_users

    # 計算量
    L_choices = np.array(params.L_candidates)
    L_u = rng.choice(L_choices, size=U)

    # アップロードサイズ（kB -> bit）
    O_choices_kB = np.array(params.O_candidates_kB)
    O_u_kB = rng.choice(O_choices_kB, size=U)
    O_u_bits = O_u_kB * 8 * 1024  # kB -> byte -> bit

    # 締切 
    D_u = rng.uniform(low=params.D_min, high=params.D_max, size=U)

    return L_u, O_u_bits, D_u


def _gen_ue_params(params: SystemParams, rng: np.random.Generator) -> np.ndarray:
    """UE送信電力 p_u を生成"""
    U = params.num_users
    p_u = rng.uniform(low=params.P_min, high=params.P_max, size=U)
    return p_u


def _gen_distance_and_channel(
    params: SystemParams,
    rng: np.random.Generator,
    U: int,
    N_MEC: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UE-MEC距離 d_um とチャネル利得 h_um, g_um, gateway を生成"""
    d_um = rng.uniform(low=params.d_min, high=params.d_max, size=(U, N_MEC))

    c = params.c
    fc = params.fc
    h_um = 4.11 * (c / (4.0 * np.pi * fc * d_um)) ** 3
    g_um = h_um.copy()  

    gateway = np.argmin(d_um, axis=1)

    return d_um, h_um, g_um, gateway


def generate_random_instance(params: SystemParams, seed: int = 1) -> Instance:
    """クラウドゲーム想定の環境に合わせてランダムなタスクを生成"""
    rng = np.random.default_rng(seed)

    U = params.num_users
    N_MEC = params.num_mecs

    L_u, O_u_bits, D_u = _gen_task_params(params, rng)
    p_u = _gen_ue_params(params, rng)
    d_um, h_um, g_um, gateway = _gen_distance_and_channel(params, rng, U, N_MEC)

    return Instance(
        L_u=L_u,
        O_u_bits=O_u_bits,
        D_u=D_u,
        p_u=p_u,
        d_um=d_um,
        h_um=h_um,
        g_um=g_um,
        gateway=gateway,
    )
