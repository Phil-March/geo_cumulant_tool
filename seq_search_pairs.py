import math
import numpy as np

def search_pairs_lag_dir_reg(lag, ndir, dimx, dimy, dimz, nlag, ltol, azm, atol, bandwh, dip, dtol, epslon, search_neib_node, vr, nx, ny, nz):
    eps_angle = 0.0000001
    min_lags = lag[0]

    for i in range(1, ndir):
        if min_lags > lag[i]:
            min_lags = lag[i]

    if dimx != 0:
        pasx = max(dimx / (nx - 1), min_lags)
    if dimy != 0:
        pasy = max(dimy / (ny - 1), min_lags)
    if dimz != 0:
        pasz = max(dimz / (nz - 1), min_lags)

    if dimx != 0:
        pasx = dimx / (nx - 1)
    if dimy != 0:
        pasy = dimy / (ny - 1)
    if dimz != 0:
        pasz = dimz / (nz - 1)
    
    max_ = nlag[0]
    nlag_accep = 0
    for loopd in range(1, ndir):
        if max_ < nlag[loopd]:
            max_ = nlag[loopd]

    if max_ < ndir:
        max_ = ndir
    max_lag = max_

    for loopd in range(ndir):
        ltol[loopd] = ltol[loopd] * lag[loopd]

    node_nlag = np.zeros((ndir, nd), dtype=int)
    node_nlag_list = np.zeros((nd, ndir, 1 * max_ + 1, 3), dtype=int)

    for loopd in range(ndir):
        azmuth = (90 - azm[loopd]) * math.pi / 180
        uvxazm[loopd] = math.cos(azmuth)
        if abs(uvxazm[loopd]) < eps_angle:
            uvxazm[loopd] = 0
        uvyazm[loopd] = math.sin(azmuth)
        if abs(uvyazm[loopd]) < eps_angle:
            uvyazm[loopd] = 0
        if atol[loopd] <= 0:
            csatol[loopd] = math.cos(45 * math.pi / 180)
        else:
            csatol[loopd] = math.cos(atol[loopd] * math.pi / 180)

        declin = (90 - dip[loopd]) * math.pi / 180
        uvzdec[loopd] = math.cos(declin)
        if abs(uvzdec[loopd]) < eps_angle:
            uvzdec[loopd] = 0
        uvhdec[loopd] = math.sin(declin)
        if abs(uvhdec[loopd]) < eps_angle:
            uvhdec[loopd] = 0
        if dtol[loopd] <= 0:
            csdtol[loopd] = math.cos(45 * math.pi / 180)
        else:
            csdtol[loopd] = math.cos(dtol[loopd] * math.pi / 180)
        dismxs[loopd] = ((nlag[loopd] + 0.5 - epslon) * lag[loopd]) ** 2

    for loopn1 in range(nd):
        p1 = loopn1
        n_voi = 0
        table = np.zeros(150, dtype=int)
        search_neib_node(loopn1, n_voi, table)

        for loopn2 in range(n_voi + 1):
            p2 = table[loopn2]
            dx = vr[p2][0] - vr[p1][0]
            dy = vr[p2][1] - vr[p1][1]
            dz = vr[p2][2] - vr[p1][2]
            dxs = dx * dx
            dys = dy * dy
            dzs = dz * dz
            hs = dxs + dys + dzs
            tes = 0

            for loopd in range(ndir):
                if hs <= dismxs[loopd]:
                    tes = 1
                    break
            if tes != 1:
                continue

            if hs <= 0:
                hs = 0
            h = math.sqrt(hs)

            for loopd in range(ndir):
                dxy = 0
                if (dxs + dys) > 0:
                    dxy = dxs + dys
                dxy = math.sqrt(dxy)
                if dxy < epslon:
                    dcazm = 1
                else:
                    dcazm = (dx * uvxazm[loopd] + dy * uvyazm[loopd]) / dxy
                if dcazm < csatol[loopd]:
                    continue

                if dcazm < 0:
                    dxy = -dxy
                if lagbeg == 1:
                    dcdec = 0
                else:
                    dcdec = (dxy * uvhdec[loopd] + dz * uvzdec[loopd]) / h
                    if dcdec < csdtol[loopd]:
                        continue

                band = uvxazm[loopd] * dy - uvyazm[loopd] * dx
                if abs(band) > bandwh[loopd]:
                    continue

                band = uvhdec[loopd] * dz - uvzdec[loopd] * dxy
                if abs(band) > bandwd[loopd]:
                    continue

                if h <= epslon:
                    lagbeg = 1
                    lagend = 1
                elif nlag[loopd] >= 2:
                    lagbeg = -1
                    lagend = -1
                    ilag = 2
                    a = abs(lag[loopd] * (ilag - 1) - ltol[loopd])
                    b = abs(lag[loopd] * (ilag - 1) + ltol[loopd])
                    if a <= h <= b:
                        if lagbeg < 0:
                            lagbeg = ilag
                        lagend = ilag
                    if lagend < 0:
                        continue

                lagend = lagbeg
                if lagend > nlag[loopd]:
                    continue

                node_nlag[loopd][p1] += 1
                node_nlag_list[p1][loopd][node_nlag[loopd][p1]][0] = lagend
                node_nlag_list[p1][loopd][node_nlag[loopd][p1]][1] = p1
                node_nlag_list[p1][loopd][node_nlag[loopd][p1]][2] = p2

                if lagend != 1:
                    for iter_lag_node in range(2, nlag[loopd] + 1):
                        vzp = 0
                        vxp = vr[p1][0] + iter_lag_node * (vr[p2][0] - vr[p1][0])
                        vyp = vr[p1][1] + iter_lag_node * (vr[p2][1] - vr[p1][1])
                        if dim == 3:
                            vzp = vr[p1][2] + iter_lag_node * (vr[p2][2] - vr[p1][2])

                        if (0 <= vxp <= dimx) and (0 <= vyp <= dimy) and (0 <= vzp <= dimz):
                            loc1 = vxp / pasx + 1
                            loc2 = vyp / pasy + 1
                            if dim == 3:
                                loc3 = vzp / pasz + 1
                            p3 = loc1 + (loc2 - 1) * nx
                            if dim == 3:
                                p3 += (loc3 - 1) * nx * ny
                            if (iter_lag_node + 1) > nlag[loopd]:
                                break
                            node_nlag[loopd][p1] += 1
                            node_nlag_list[p1][loopd][node_nlag[loopd][p1]][0] = iter_lag_node + 1
                            node_nlag_list[p1][loopd][node_nlag[loopd][p1]][1] = p1
                            node_nlag_list[p1][loopd][node_nlag[loopd][p1]][2] = p3 - 1

    return node_nlag, node_nlag_list
