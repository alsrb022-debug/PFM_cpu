import configparser
import math
import time
from pathlib import Path

import numpy as np

from init_benchmark import (
    init_plane,
    init_triple,
    init_circle,
    init_sphere,
    init_neighbors,
)
from init_voronoi import init_voronoi

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


nmax = None
EMPTY_ID = -1


# ============================================================
# Main
# ============================================================
def run(input_path="input.txt", out_dir="pout"):
    p = read_input(input_path)
    out_dir = prepare_output_dir(out_dir)
    params = build_params(p, out_dir)

    xi, www, cep, emm, ndim = compute_pf_coefficients(params)

    im = params["im"]
    jm = params["jm"]
    km = params["km"]

    dx = params["dx"]
    dy = params["dy"]
    dz = params["dz"]

    pss = params["pss"]
    tifac = params["tifac"]

    nstep = params["nstep"]
    nout = params["nout"]

    init_type = params["init_type"]
    bc_type = params["bc_type"]

    seed = params.get("seed", 1234)
    nphase = params.get("nphase", 0)
    radius = params.get("radius", 0.0)
    cx = params.get("cx", -1.0)
    cy = params.get("cy", -1.0)
    cz = params.get("cz", -1.0)
    ngrain = params.get("ngrain", 0)

    cepkl = params["cepkl"]
    wwwkl = params["wwwkl"]
    emmkl = params["emmkl"]

    phiO = np.zeros((im + 2, jm + 2, km + 2, nmax), dtype=np.float64)
    phiN = np.zeros((im + 2, jm + 2, km + 2, nmax), dtype=np.float64)

    idO = np.full((im + 2, jm + 2, km + 2, nmax), EMPTY_ID, dtype=np.int32)
    idN = np.full((im + 2, jm + 2, km + 2, nmax), EMPTY_ID, dtype=np.int32)

    initialize_fields(
        phiN, idN,
        im, jm, km,
        init_type, bc_type,
        seed=seed,
        nphase=nphase,
        radius=radius,
        cx=cx, cy=cy, cz=cz,
        ngrain=ngrain,
    )

    print("Simulation started...")
    if NUMBA_OK:
        print("Numba JIT: enabled")
    else:
        print("Numba JIT: disabled")

    print(f"init = {init_type}")
    print(f"bc   = {bc_type}")

    sim_time = 0.0
    t0 = time.perf_counter()

    for nnn in range(0, nstep + 1):
        dt = compute_dt(dx, cep, emm, ndim, tifac)
        sim_time += dt

        phiO[:, :, :, :] = phiN[:, :, :, :]
        idO[:, :, :, :] = idN[:, :, :, :]

        apply_bc(phiO, idO, bc_type, im, jm, km)

        evolve_one_step(
            phiO, phiN,
            idO, idN,
            im, jm, km,
            dx, dy, dz, dt, pss,
            cep, www, emm,
            cepkl, wwwkl, emmkl
        )

        if nnn % nout == 0:
            cpu_elapsed = time.perf_counter() - t0
            print(f"Step: {nnn:8d}  Time: {sim_time:.6e}  CPU: {int(cpu_elapsed)} s")
            write_snapshot(phiN, idN, im, jm, km, out_dir, nnn, sim_time)

    print("Simulation successfully complete.")


# ============================================================
# Input
# ============================================================
def read_input(path="input.txt"):
    global nmax

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(path, encoding="utf-8")

    data = {}

    data["im"] = cfg.getint("domain", "im")
    data["jm"] = cfg.getint("domain", "jm")
    data["km"] = cfg.getint("domain", "km")

    data["dx"] = cfg.getfloat("grid", "dx")
    data["dy"] = cfg.getfloat("grid", "dy")
    data["dz"] = cfg.getfloat("grid", "dz")

    nmax = cfg.getint("pf", "nmax")
    data["emob"] = cfg.getfloat("pf", "emob")
    data["sigma"] = cfg.getfloat("pf", "sigma")
    data["xi_in"] = cfg.getfloat("pf", "xi_in")
    data["tifac"] = cfg.getfloat("pf", "tifac")
    data["pss"] = cfg.getfloat("pf", "pss")

    data["nstep"] = cfg.getint("time", "nstep")
    data["nout"] = cfg.getint("time", "nout")

    init_type = cfg.get("init", "type", fallback="voronoi").strip().lower()
    data["init_type"] = init_type

    if init_type == "voronoi":
        data["seed"] = cfg.getint("init", "seed", fallback=1234)
        data["nphase"] = cfg.getint("init", "nphase")

    elif init_type == "plane":
        pass

    elif init_type == "triple":
        pass

    elif init_type == "circle":
        data["radius"] = cfg.getfloat("init", "radius")
        data["cx"] = cfg.getfloat("init", "cx", fallback=-1.0)
        data["cy"] = cfg.getfloat("init", "cy", fallback=-1.0)

    elif init_type == "sphere":
        data["radius"] = cfg.getfloat("init", "radius")
        data["cx"] = cfg.getfloat("init", "cx", fallback=-1.0)
        data["cy"] = cfg.getfloat("init", "cy", fallback=-1.0)
        data["cz"] = cfg.getfloat("init", "cz", fallback=-1.0)

    elif init_type == "neighbors":
        data["ngrain"] = cfg.getint("init", "ngrain")

    else:
        raise ValueError(f"Unsupported init type: {init_type}")

    data["bc_type"] = cfg.get("bc", "type", fallback="pbc").strip().lower()

    return data


# ============================================================
# Setup
# ============================================================
def prepare_output_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_params(p, out_dir):
    params = {}

    params["im"] = p["im"]
    params["jm"] = p["jm"]
    params["km"] = p["km"]

    params["dx"] = p["dx"]
    params["dy"] = p["dy"]
    params["dz"] = p["dz"]

    params["emob"] = p["emob"]
    params["tifac"] = p["tifac"]
    params["sigma"] = p["sigma"]
    params["xi_in"] = p["xi_in"]
    params["pss"] = p["pss"]

    params["nstep"] = p["nstep"]
    params["nout"] = p["nout"]

    params["init_type"] = p["init_type"]
    params["bc_type"] = p["bc_type"]

    if "seed" in p:
        params["seed"] = p["seed"]
    if "nphase" in p:
        params["nphase"] = p["nphase"]
    if "radius" in p:
        params["radius"] = p["radius"]
    if "cx" in p:
        params["cx"] = p["cx"]
    if "cy" in p:
        params["cy"] = p["cy"]
    if "cz" in p:
        params["cz"] = p["cz"]
    if "ngrain" in p:
        params["ngrain"] = p["ngrain"]

    params["out_dir"] = out_dir

    params["cepkl"] = np.zeros((3, 3), dtype=np.float64)
    params["wwwkl"] = np.zeros((3, 3), dtype=np.float64)
    params["emmkl"] = np.zeros((3, 3), dtype=np.float64)

    return params


def compute_pf_coefficients(params):
    pi = math.pi

    dx = params["dx"]
    sigma = params["sigma"]
    emob = params["emob"]
    xi_in = params["xi_in"]

    xi = xi_in * dx
    www = 2.0 * sigma / xi
    cep = 4.0 / pi * math.sqrt(xi * sigma)
    emm = emob * sigma / (cep ** 2.0)

    ndim = 0
    if params["im"] > 1:
        ndim += 1
    if params["jm"] > 1:
        ndim += 1
    if params["km"] > 1:
        ndim += 1

    return xi, www, cep, emm, ndim


def compute_dt(dx, cep, emm, ndim, tifac):
    return dx ** 2.0 / (2.0 * ndim * (cep ** 2.0 * emm)) * tifac


# ============================================================
# Initialization
# ============================================================
def initialize_fields(
    phiN, idN,
    im, jm, km,
    init_type, bc_type,
    seed=1234,
    nphase=0,
    radius=0.0,
    cx=-1.0, cy=-1.0, cz=-1.0,
    ngrain=0,
):
    if init_type == "voronoi":
        init_voronoi(phiN, idN, im, jm, km, nphase, seed, bc_type)

    elif init_type == "plane":
        init_plane(phiN, idN, im, jm, km)

    elif init_type == "triple":
        init_triple(phiN, idN, im, jm, km)

    elif init_type == "circle":
        cx_use = None if cx < 0.0 else cx
        cy_use = None if cy < 0.0 else cy
        init_circle(phiN, idN, im, jm, km, radius, cx_use, cy_use)

    elif init_type == "sphere":
        cx_use = None if cx < 0.0 else cx
        cy_use = None if cy < 0.0 else cy
        cz_use = None if cz < 0.0 else cz
        init_sphere(phiN, idN, im, jm, km, radius, cx_use, cy_use, cz_use)

    elif init_type == "neighbors":
        init_neighbors(phiN, idN, im, jm, km, ngrain)

    else:
        raise ValueError(f"Unsupported init type: {init_type}")


# ============================================================
# Output
# ============================================================
def write_snapshot(phiN, idN, im, jm, km, out_dir, nnn, sim_time):
    phi_save = phiN[1:im + 1, 1:jm + 1, 1:km + 1, :].copy()
    id_save = idN[1:im + 1, 1:jm + 1, 1:km + 1, :].copy()

    np.savez_compressed(
        out_dir / f"p_{nnn:08d}.npz",
        phi=phi_save,
        id=id_save,
        step=np.int32(nnn),
        time=np.float64(sim_time),
    )


# ============================================================
# Boundary conditions
# ============================================================
def apply_bc(phiO, idO, bc_type, im, jm, km):
    if bc_type == "pbc":
        apply_bc_pbc(phiO, idO, im, jm, km)
    elif bc_type == "neumann":
        apply_bc_neumann(phiO, idO, im, jm, km)
    else:
        raise ValueError(f"Unsupported bc type: {bc_type}")


@njit(cache=True)
def apply_bc_pbc(phiO, idO, im, jm, km):
    # z
    for i in range(im + 2):
        for j in range(jm + 2):
            phiO[i, j, 0, :] = phiO[i, j, km, :]
            phiO[i, j, km + 1, :] = phiO[i, j, 1, :]
            idO[i, j, 0, :] = idO[i, j, km, :]
            idO[i, j, km + 1, :] = idO[i, j, 1, :]

    # y
    for i in range(im + 2):
        for k in range(km + 2):
            phiO[i, 0, k, :] = phiO[i, jm, k, :]
            phiO[i, jm + 1, k, :] = phiO[i, 1, k, :]
            idO[i, 0, k, :] = idO[i, jm, k, :]
            idO[i, jm + 1, k, :] = idO[i, 1, k, :]

    # x
    for j in range(jm + 2):
        for k in range(km + 2):
            phiO[0, j, k, :] = phiO[im, j, k, :]
            phiO[im + 1, j, k, :] = phiO[1, j, k, :]
            idO[0, j, k, :] = idO[im, j, k, :]
            idO[im + 1, j, k, :] = idO[1, j, k, :]


@njit(cache=True)
def apply_bc_neumann(phiO, idO, im, jm, km):
    # z
    for i in range(im + 2):
        for j in range(jm + 2):
            phiO[i, j, 0, :] = phiO[i, j, 1, :]
            phiO[i, j, km + 1, :] = phiO[i, j, km, :]
            idO[i, j, 0, :] = idO[i, j, 1, :]
            idO[i, j, km + 1, :] = idO[i, j, km, :]

    # y
    for i in range(im + 2):
        for k in range(km + 2):
            phiO[i, 0, k, :] = phiO[i, 1, k, :]
            phiO[i, jm + 1, k, :] = phiO[i, jm, k, :]
            idO[i, 0, k, :] = idO[i, 1, k, :]
            idO[i, jm + 1, k, :] = idO[i, jm, k, :]

    # x
    for j in range(jm + 2):
        for k in range(km + 2):
            phiO[0, j, k, :] = phiO[1, j, k, :]
            phiO[im + 1, j, k, :] = phiO[im, j, k, :]
            idO[0, j, k, :] = idO[1, j, k, :]
            idO[im + 1, j, k, :] = idO[im, j, k, :]


# ============================================================
# Sparse helper
# ============================================================
@njit(cache=True)
def get_phi_at(phi, id_arr, i, j, k, gid):
    for s in range(nmax):
        if id_arr[i, j, k, s] == gid:
            return phi[i, j, k, s]
    return 0.0


@njit(cache=True)
def append_gid_if_new(cand_ids, count, gid):
    if gid == EMPTY_ID:
        return count

    for t in range(count):
        if cand_ids[t] == gid:
            return count

    if count < nmax:
        cand_ids[count] = gid
        return count + 1

    return count


# ============================================================
# One time step
# ============================================================
@njit(cache=True)
def evolve_one_step(
    phiO, phiN,
    idO, idN,
    im, jm, km,
    dx, dy, dz, dt, pss,
    cep, www, emm,
    cepkl, wwwkl, emmkl
):
    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    cand_ids = np.empty(nmax, dtype=np.int32)
    df = np.empty(nmax, dtype=np.float64)
    tmp_phi = np.empty(nmax, dtype=np.float64)

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                for s in range(nmax):
                    cand_ids[s] = EMPTY_ID
                    df[s] = 0.0
                    tmp_phi[s] = 0.0

                count = 0

                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i, j, k, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i + 1, j, k, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i - 1, j, k, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i, j + 1, k, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i, j - 1, k, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i, j, k + 1, s])
                for s in range(nmax):
                    count = append_gid_if_new(cand_ids, count, idO[i, j, k - 1, s])

                nph = 0

                for kk in range(count):
                    gid_k = cand_ids[kk]

                    pk0 = get_phi_at(phiO, idO, i, j, k, gid_k)
                    pk1 = get_phi_at(phiO, idO, i + 1, j, k, gid_k)
                    pk2 = get_phi_at(phiO, idO, i - 1, j, k, gid_k)
                    pk3 = get_phi_at(phiO, idO, i, j + 1, k, gid_k)
                    pk4 = get_phi_at(phiO, idO, i, j - 1, k, gid_k)
                    pk5 = get_phi_at(phiO, idO, i, j, k + 1, gid_k)
                    pk6 = get_phi_at(phiO, idO, i, j, k - 1, gid_k)

                    pksum = pk0 + pk1 + pk2 + pk3 + pk4 + pk5 + pk6
                    Krange = 0

                    if pksum >= pss:
                        nph += 1
                        if pksum <= (7.0 - pss):
                            Krange = 1

                    for ll in range(count):
                        gid_l = cand_ids[ll]
                        if gid_l == gid_k:
                            continue

                        pl0 = get_phi_at(phiO, idO, i, j, k, gid_l)
                        pl1 = get_phi_at(phiO, idO, i + 1, j, k, gid_l)
                        pl2 = get_phi_at(phiO, idO, i - 1, j, k, gid_l)
                        pl3 = get_phi_at(phiO, idO, i, j + 1, k, gid_l)
                        pl4 = get_phi_at(phiO, idO, i, j - 1, k, gid_l)
                        pl5 = get_phi_at(phiO, idO, i, j, k + 1, gid_l)
                        pl6 = get_phi_at(phiO, idO, i, j, k - 1, gid_l)

                        plsum = pl0 + pl1 + pl2 + pl3 + pl4 + pl5 + pl6
                        Lrange = 0

                        if plsum >= pss:
                            if plsum <= (7.0 - pss):
                                Lrange = 1

                        if Krange == 1 and Lrange == 1:
                            cep_loc = cep
                            www_loc = www

                            phixxl = (pl1 - 2.0 * pl0 + pl2) / dx2
                            phiyyl = (pl3 - 2.0 * pl0 + pl4) / dy2
                            phizzl = (pl5 - 2.0 * pl0 + pl6) / dz2

                            df[kk] += 0.5 * cep_loc * cep_loc * (phixxl + phiyyl + phizzl) + www_loc * pl0

                for kk in range(count):
                    gid_k = cand_ids[kk]
                    pk0 = get_phi_at(phiO, idO, i, j, k, gid_k)

                    plkk = 0.0

                    for ll in range(count):
                        gid_l = cand_ids[ll]
                        if gid_l == gid_k:
                            continue

                        emm_loc = emm
                        plkk += emm_loc * (df[kk] - df[ll])

                    if nph > 0:
                        val = pk0 + dt * (-2.0 / float(nph) * plkk)
                    else:
                        val = pk0

                    # 0보다 작은 값만 잘라냄
                    if val < 0.0:
                        val = 0.0

                    tmp_phi[kk] = val

                pNsum = 0.0
                for kk in range(count):
                    pNsum += tmp_phi[kk]

                if pNsum > 0.0:
                    inv = 1.0 / pNsum
                    for kk in range(count):
                        tmp_phi[kk] *= inv
                else:
                    for kk in range(count):
                        gid_k = cand_ids[kk]
                        tmp_phi[kk] = get_phi_at(phiO, idO, i, j, k, gid_k)

                # tmp_phi 큰 값 순서로 정렬, cand_ids도 같이 이동
                for a in range(count - 1):
                    max_idx = a
                    for b in range(a + 1, count):
                        if tmp_phi[b] > tmp_phi[max_idx]:
                            max_idx = b

                    if max_idx != a:
                        tmp_val = tmp_phi[a]
                        tmp_phi[a] = tmp_phi[max_idx]
                        tmp_phi[max_idx] = tmp_val

                        tmp_id = cand_ids[a]
                        cand_ids[a] = cand_ids[max_idx]
                        cand_ids[max_idx] = tmp_id

                for s in range(nmax):
                    phiN[i, j, k, s] = 0.0
                    idN[i, j, k, s] = EMPTY_ID

                slot = 0
                for kk in range(count):
                    if tmp_phi[kk] >= pss and slot < nmax:
                        phiN[i, j, k, slot] = tmp_phi[kk]
                        idN[i, j, k, slot] = cand_ids[kk]
                        slot += 1

                if slot == 0 and count > 0:
                    best_k = 0
                    best_phi = tmp_phi[0]
                    for kk in range(1, count):
                        if tmp_phi[kk] > best_phi:
                            best_phi = tmp_phi[kk]
                            best_k = kk

                    phiN[i, j, k, 0] = 1.0
                    idN[i, j, k, 0] = cand_ids[best_k]

if __name__ == "__main__":
    run("input.txt", out_dir="p_out")
