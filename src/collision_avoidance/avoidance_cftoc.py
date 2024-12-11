"""
12-9-24 By: George Riera

Collision avoidance controller for states [x, y, theta] and inputs [v, w].
    
    Parameters:
        x0: Initial state [x, y, theta].
        obstacles: List of obstacles, each defined as a tuple (x_obs, y_obs, r_obs).
        N: Prediction horizon.
        Ts: Sampling time.
        v_max, w_max: Maximum linear and angular velocities.
        x_ref, y_ref: Reference trajectory points.
        R, Q: Control and tracking weights.
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from scipy.interpolate import CubicSpline


def cftoc_collision_avoidance(z0, obstacles, N, Ts, v_max, w_max, x_ref, y_ref,R, Q):

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
"""
# Spline interpolation for reference trajectory
# Waypoints
waypoints_x = [0, 2, 4, 6]
waypoints_y = [0, 1, 1, 0]

# Time points
t = np.linspace(0, 1, len(waypoints_x))

# Interpolate
cs_x = CubicSpline(t, waypoints_x)
cs_y = CubicSpline(t, waypoints_y)

# Generate reference points
t_dense = np.linspace(0, 1, 100)  # Dense time array
x_ref = cs_x(t_dense)
y_ref = cs_y(t_dense)
"""

# Test-------------------------------------------------------------
# Parameters
z0 = [0,0,0] # Initial state
obstacles = []
N = 20  # Prediction horizon
Ts = 0.1 # Sampling time
v_max = 1.0
w_max = 0.5
R = 1
Q = 10 

# Define a straight Reference trajectory 

x_ref = np.linspace(0, 5, N+1)
y_ref = np.linspace(0,5, N+1)
x_traj, u_traj, cost = cftoc_collision_avoidance(z0, obstacles, N, Ts, v_max, w_max, x_ref, y_ref, R, Q)

# Plot 
# Plot trajectory
plt.plot(x_traj[0, :], x_traj[1, :], label="Robot Trajectory", marker='o')
plt.plot(x_ref, y_ref, label="Reference Trajectory", linestyle='--')

# Labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.title("Collision Avoidance Trajectory")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()