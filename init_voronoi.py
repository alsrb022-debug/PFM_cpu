import numpy as np

EMPTY_ID = -1


def init_voronoi(phiN, idN, im, jm, km, nphase, seed, bc_type):
    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    rng = np.random.default_rng(seed)

    rx = rng.random(nphase) * im + 1.0
    ry = rng.random(nphase) * jm + 1.0
    rz = rng.random(nphase) * km + 1.0

    use_pbc = (bc_type == "pbc")

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                ri = float(i)
                rj = float(j)
                rk = float(k)

                rmin = 1.0e100
                nmin = 0

                for n in range(nphase):
                    dx = abs(rx[n] - ri)
                    dy = abs(ry[n] - rj)
                    dz = abs(rz[n] - rk)

                    if use_pbc:
                        dx = min(dx, im - dx)
                        dy = min(dy, jm - dy)
                        dz = min(dz, km - dz)

                    if jm == 1:
                        dy = 0.0
                    if km == 1:
                        dz = 0.0

                    rr = dx * dx + dy * dy + dz * dz

                    if rr < rmin:
                        rmin = rr
                        nmin = n

                phiN[i, j, k, 0] = 1.0
                idN[i, j, k, 0] = nmin
