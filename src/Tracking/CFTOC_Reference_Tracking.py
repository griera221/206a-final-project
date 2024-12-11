"""
12-10-24 By: George Riera

"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

def cftoc_trajectory(z0, obstacles, N, Ts, v_max, w_max, x_ref, y_ref,Xf={'type':''}):

    nx = 3
    nu = 2

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0,N+1))   # length of finitie optimization problem
    model.tidu = pyo.Set(initialize=range(0,N)) # length of finite optimization problem
    model.xidx = pyo.Set(initialize=range(0,nx))
    model.uidx = pyo.Set(initialize=range(0,nu))

    # Create state and input bariables trajectory
    model.z = pyo.Var(model.xidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidu)

    # Tuning
    model.R = 2
    model.Q = 1
    
    # Define polytope terminal constraint
    if Xf['type']=='polytope':
        model.nf = np.size(Xf['Af'], 0)
        model.nfidx= pyo.Set( initialize= range(model.nf), ordered=True )
        model.Af = Xf['Af']
        model.bf = Xf['bf']

     # Cost function (tracking error + control effort)
    model.ref_tracking = sum(
            model.Q * ((model.z[0, t] - x_ref[t])**2 + (model.z[1, t] - y_ref[t])**2) +
            model.R * (model.u[0, t]**2 + model.u[1, t]**2)
            for t in model.tidu
        )
    model.cost = pyo.Objective(expr = model.ref_tracking, sense=pyo.minimize)
     

    # Initial Constraint
    model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i, 0] == z0[i])

    # Dynamic constraint
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] == model.z[0, t] + Ts* (pyo.cos(model.z[2, t]) * model.u[0,t])
                                       if t < N else pyo.Constraint.Skip)
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] == model.z[1, t] + Ts* (pyo.sin(model.z[2, t]) *model.u[0, t])
                                   if t < N else pyo.Constraint.Skip)
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] == model.z[2, t] + Ts* model.u[1, t]
                                   if t < N else pyo.Constraint.Skip)
    
    # Input Constraints
    model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= 2
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >=-2
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= 2
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= -2
                                   if t <= N-1 else pyo.Constraint.Skip)


    # Final Constraint
    if Xf['type'] == 'polytope':
        def final_const_rule(model, i):
            return sum(model.Af[i,j] * model.z[j,N] for j in model.xidx) <= model.bf[i]
        model.final_const = pyo.Constraint(model.nfidx, rule=final_const_rule)
    #model.constraint9 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i,N] == zf[i])

    # Solve 
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError("Solver did not find an optimal solution.")

    # Extract results
    x_traj = np.array([[pyo.value(model.z[i, t]) for t in model.tidx] for i in range(nx)])
    u_traj = np.array([[pyo.value(model.u[i, t]) for t in model.tidu] for i in range(nu)])
    cost = pyo.value(model.cost)

    return x_traj, u_traj, cost