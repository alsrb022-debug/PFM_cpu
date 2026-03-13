import math
import numpy as np

EMPTY_ID = -1


def init_plane(phiN, idN, im, jm, km):
    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    ic = (im + 1) // 2

    for i in range(1, im + 1):
        gid = 0 if i < ic else 1

        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                phiN[i, j, k, 0] = 1.0
                idN[i, j, k, 0] = gid


def init_triple(phiN, idN, im, jm, km):
    if km != 1:
        raise ValueError("triple init is for 2D only (use km = 1)")

    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    cx = 0.5 * (im + 1)
    cy = 0.5 * (jm + 1)

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            x = float(i) - cx
            y = float(j) - cy
            th = math.atan2(y, x)

            if -math.pi / 3.0 <= th < math.pi / 3.0:
                gid = 0
            elif math.pi / 3.0 <= th or th < -math.pi:
                gid = 1
            else:
                gid = 2

            phiN[i, j, 1, 0] = 1.0
            idN[i, j, 1, 0] = gid


def init_circle(phiN, idN, im, jm, km, radius, cx=None, cy=None):
    if km != 1:
        raise ValueError("circle init is for 2D only (use km = 1)")

    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    if cx is None:
        cx = 0.5 * (im + 1)
    if cy is None:
        cy = 0.5 * (jm + 1)

    r2 = radius * radius

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            dx = float(i) - cx
            dy = float(j) - cy

            gid = 1 if (dx * dx + dy * dy) <= r2 else 0

            phiN[i, j, 1, 0] = 1.0
            idN[i, j, 1, 0] = gid


def init_sphere(phiN, idN, im, jm, km, radius, cx=None, cy=None, cz=None):
    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    if cx is None:
        cx = 0.5 * (im + 1)
    if cy is None:
        cy = 0.5 * (jm + 1)
    if cz is None:
        cz = 0.5 * (km + 1)

    r2 = radius * radius

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            for k in range(1, km + 1):
                dx = float(i) - cx
                dy = float(j) - cy
                dz = float(k) - cz

                gid = 1 if (dx * dx + dy * dy + dz * dz) <= r2 else 0

                phiN[i, j, k, 0] = 1.0
                idN[i, j, k, 0] = gid


def init_neighbors(phiN, idN, im, jm, km, ngrain):
    if km != 1:
        raise ValueError("neighbors init is for 2D only (use km = 1)")

    phiN[:, :, :, :] = 0.0
    idN[:, :, :, :] = EMPTY_ID

    # circle과 똑같은 중심 정의
    cx = 0.5 * (im + 1)
    cy = 0.5 * (jm + 1)

    # 중앙 입자 반지름: 필요하면 여기 숫자만 바꾸면 됨
    radius = 20.0
    r2 = radius * radius

    for i in range(1, im + 1):
        for j in range(1, jm + 1):
            dx = float(i) - cx
            dy = float(j) - cy
            rr = dx * dx + dy * dy

            # 1) 먼저 circle처럼 중앙 입자 생성
            if rr <= r2:
                gid = 0

            # 2) 바깥 영역은 neighbors처럼 분할
            else:
                th = math.atan2(dy, dx)
                if th < 0.0:
                    th += 2.0 * math.pi

                sector = int(ngrain * th / (2.0 * math.pi))
                if sector >= ngrain:
                    sector = ngrain - 1

                # 바깥 grain은 1 ~ ngrain
                gid = sector + 1

            phiN[i, j, 1, 0] = 1.0
            idN[i, j, 1, 0] = gid
