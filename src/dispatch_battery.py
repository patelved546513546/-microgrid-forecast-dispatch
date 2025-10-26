# Fallback-friendly battery optimizer.
# Tries CVXPY with any installed solver (SCS/ECOS). If none, uses a greedy heuristic.

from typing import Dict
import numpy as np

try:
    import cvxpy as cp
except Exception:
    cp = None  # allow fallback without cvxpy

def _solve_with_cvxpy(netload_kw, price_rs_per_kwh, dt_hours,
                      e_cap_kwh, p_cap_kw, eta_c, eta_d, soc0,
                      demand_charge_rs_per_kw) -> Dict:
    if cp is None:
        raise RuntimeError("cvxpy not available")

    T = len(netload_kw)
    c = cp.Variable(T, nonneg=True)   # charge power (kW)
    d = cp.Variable(T, nonneg=True)   # discharge power (kW)
    soc = cp.Variable(T+1)            # state of charge (kWh)
    grid = cp.Variable(T, nonneg=True) # grid import (kW)
    peak = cp.Variable(nonneg=True)

    constraints = [soc[0] == soc0 * e_cap_kwh]
    for t in range(T):
        constraints += [
            soc[t+1] == soc[t] + (eta_c*c[t] - d[t]/eta_d) * dt_hours,
            0 <= soc[t+1], soc[t+1] <= e_cap_kwh,
            0 <= c[t], c[t] <= p_cap_kw,
            0 <= d[t], d[t] <= p_cap_kw,
            grid[t] >= netload_kw[t] + c[t] - d[t],  # import only (no export)
            grid[t] >= 0,
            peak >= grid[t],
        ]
    constraints += [soc[T] == soc[0]]  # end the day where we started

    energy_cost = cp.sum(cp.multiply(price_rs_per_kwh * dt_hours, grid))
    demand_cost = demand_charge_rs_per_kw * peak
    prob = cp.Problem(cp.Minimize(energy_cost + demand_cost), constraints)

    # Pick the first available solver
    solver_name = None
    if hasattr(cp, "installed_solvers"):
        for s in ["SCS", "ECOS"]:
            if s in cp.installed_solvers():
                solver_name = s
                break
    if solver_name is None:
        raise RuntimeError("No suitable CVXPY solver (SCS/ECOS) installed.")

    prob.solve(solver=getattr(cp, solver_name), verbose=False)

    return dict(
        grid=grid.value, charge=c.value, discharge=d.value, soc=soc.value[:-1],
        peak=peak.value, status=f"cvxpy-{solver_name}", cost=float(energy_cost.value + demand_cost.value)
    )

def _greedy_fallback(netload_kw, price_rs_per_kwh, dt_hours,
                     e_cap_kwh, p_cap_kw, eta_c, eta_d, soc0,
                     demand_charge_rs_per_kw) -> Dict:
    # Simple peak-shave + price arbitrage
    nl = np.asarray(netload_kw, dtype=float).copy()
    prices = np.asarray(price_rs_per_kwh, dtype=float).copy()
    T = len(nl)

    charge = np.zeros(T)
    discharge = np.zeros(T)
    grid = np.zeros(T)
    soc = np.zeros(T+1)
    soc[0] = soc0 * e_cap_kwh

    low_th = np.quantile(prices, 0.35)
    high_th = np.quantile(prices, 0.70)
    target_peak = 0.9 * np.max(nl)  # shave ~10%

    for t in range(T):
        # max power limited by SOC and capacity
        d_max_soc = min(p_cap_kw, soc[t] * eta_d / dt_hours)  # ensure soc[t+1] >= 0
        c_max_soc = min(p_cap_kw, (e_cap_kwh - soc[t]) / (eta_c * dt_hours))

        d = 0.0; c = 0.0

        # 1) Peak shaving has priority
        if nl[t] > target_peak and d_max_soc > 0:
            d = min(d_max_soc, nl[t] - target_peak)

        # 2) Price arbitrage if not shaving peak
        if d == 0.0:
            if prices[t] <= low_th and c_max_soc > 0:
                c = c_max_soc
            elif prices[t] >= high_th and d_max_soc > 0:
                d = d_max_soc

        # Avoid exporting power (grid import should be >= 0)
        d = min(d, nl[t] + c)

        # Apply state transition
        soc[t+1] = soc[t] + (eta_c * c - d / eta_d) * dt_hours
        soc[t+1] = min(max(soc[t+1], 0.0), e_cap_kwh)

        charge[t] = c
        discharge[t] = d
        grid[t] = max(0.0, nl[t] + c - d)

    energy_cost = float(np.sum(prices * dt_hours * grid))
    peak = float(np.max(grid))
    demand_cost = demand_charge_rs_per_kw * peak
    cost = energy_cost + demand_cost

    return dict(
        grid=grid, charge=charge, discharge=discharge, soc=soc[:-1],
        peak=peak, status="greedy-fallback", cost=cost
    )

def optimize_battery(netload_kw, price_rs_per_kwh, dt_hours=0.5,
                     e_cap_kwh=50, p_cap_kw=20, eta_c=0.95, eta_d=0.95,
                     soc0=0.5, demand_charge_rs_per_kw=300) -> Dict:
    netload_kw = np.asarray(netload_kw, dtype=float)
    price_rs_per_kwh = np.asarray(price_rs_per_kwh, dtype=float)

    # Try CVXPY solver first
    try:
        return _solve_with_cvxpy(netload_kw, price_rs_per_kwh, dt_hours,
                                 e_cap_kwh, p_cap_kw, eta_c, eta_d, soc0,
                                 demand_charge_rs_per_kw)
    except Exception:
        # Fallback that requires no external solver
        return _greedy_fallback(netload_kw, price_rs_per_kwh, dt_hours,
                                e_cap_kwh, p_cap_kw, eta_c, eta_d, soc0,
                                demand_charge_rs_per_kw)