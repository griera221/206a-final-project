"""
12-9-24 By: George Riera

"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import polytope as pt



# Paramaters
Ts = 0.1 #seconds = 10 samples per second
N = 50
Tfinal = Ts*N

# Initial and final position
z0 = [75,25,0]
zf = [125,55,np.pi/2]

# Setup the CTFOC problem with pyomo

def cftoc_tracking(z0,zf,N,Ts, Xf={'type':''},polytopes=[]):

    nx = 3         # number of states (x,y,theta)
    nu = 2         # number of inputs u1=v, u2=omega

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0, N+1))      # Length of finite optimization prblm
    model.xidx = pyo.Set(initialize=range(0,nx))
    model.uidx = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.z = pyo.Var(model.xidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)
    """
    if Xf['type']=='polytope':
        model.nf = np.size(Xf['Af'], 0)
        model.nfidx= pyo.Set( initialize= range(model.nf), ordered=True )
        model.Af = Xf['Af']
        model.bf = Xf['bf']

        def final_const_rule(model, i):
            return sum(model.Af[i,j]*model.z[j,N] for j in model.xidx) <= model.bf[i]
        model.final_const = pyo.Constraint(model.nfidx, rule=final_const_rule)
    """
    # Terminal Set Constraint
    

    # Objective function

    model.shortest_time_cost = sum((model.z[1,t]-zf[1])**2 for t in model.tidx if t<N)
    #model.shortest_time_cost = sum(
    #    (model.z[0, t])**2 + (model.z[1,t])**2 + (model.z[2,t])**2 
    #    for t in model.tidx if t<N
    #    )
    model.cost = pyo.Objective(expr = model.shortest_time_cost, sense = pyo.minimize)
    # Constraint list

    # Obstacle avoidance constraints
    for poly_idx, poly in enumerate(polytopes):
        A_obs = poly.A
        b_obs = poly.b
        n_obs = A_obs.shape[0]

        def obstacle_avoidance_rule(model, t, j):
            if t == N:
                return pyo.Constraint.Skip # Skip obstacle constraints at the final timestep
            return sum(A_obs[j,i]*model.z[i,t] for i in range(2)) >= b_obs[j]
        
        model.add_component(
            f"obstacle_avoidance_{poly_idx}",
            pyo.Constraint(model.tidx, range(n_obs), rule=obstacle_avoidance_rule)
        )

    def verify_obstacle_constraints(model, polytopes):
        for poly_idx, poly in enumerate(polytopes):
            A_obs = poly.A
            b_obs = poly.b
            n_obs = A_obs.shape[0]

            print(f"Obstacle {poly_idx}:")
            for t in model.tidx:
                if t == N:  # Skip final time step
                    continue
                for j in range(n_obs):
                    lhs = sum(A_obs[j, i] * pyo.value(model.z[i, t]) for i in range(2))
                    rhs = b_obs[j]
                    print(f"  t={t}, facet={j}: {lhs} >= {rhs} -> {'PASS' if lhs >= rhs else 'FAIL'}")
    """
    # Define obstacle constraints 
    obstacles = [
    {'x_min': 0, 'x_max': 1, 'y_min': 2, 'y_max': 3},  # (1,1)
    {'x_min': 2, 'x_max': 3, 'y_min': 2, 'y_max': 3},  # (1,3)
    {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1},  # (3,1)
    {'x_min': 2, 'x_max': 3, 'y_min': 0, 'y_max': 1}   # (3,3)
    ]

    def obstacle_avoidance(model, t):
        constraints = []
        for obs in obstacles:
            # Each obstacle creates constraints
            constraints.append(model.x[t] <= obs['x_min'])
            constraints.append(model.x[t] <= obs['x_max'])
            constraints.append(model.y[t] <= obs['x_max'])
            constraints.append(model.y[t] <= obs['x_max'])
        return sum(constraints) >= 1 # At least one constraint satisfied to avoid region
    
    model.obstacle_avoid = pyo.Constraint(model.tidx, rule=obstacle_avoidance)
    """
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
    model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= 5
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >=-5
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= 5
                                   if t <= N-1 else pyo.Constraint.Skip)
    model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= -5
                                   if t <= N-1 else pyo.Constraint.Skip)


    # Final Constraint

    #model.constraint9 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i,N] == zf[i])

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model,tee=True)
    if (results.solver.status != pyo.SolverStatus.ok) or (results.solver.termination_condition != pyo.TerminationCondition.optimal):
        print("Solver failed!")
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")

    # Debugging: Verify obstacle constraints
    verify_obstacle_constraints(model, polytopes)

    # Store solver results
    z1 = [pyo.value(model.z[0,0])]
    z2 = [pyo.value(model.z[1,0])]
    z3 = [pyo.value(model.z[2,0])]
    u1 = [pyo.value(model.z[0,0])]
    u2 = [pyo.value(model.z[1,0])]

    for t in model.tidx:
        if t<N:
            z1.append(pyo.value(model.z[0,t+1]))
            z2.append(pyo.value(model.z[1,t+1]))
            z3.append(pyo.value(model.z[2,t+1]))
        if t < N-1:
            u1.append(pyo.value(model.u[0,t+1]))
            u1.append(pyo.value(model.u[1,t+1]))

    return z1,z2,z3,u1,u2 



# Create polytopes for obstacles

# Define list of bounds
obstacles = [
    {'x_min': 0, 'x_max': 50, 'y_min': 100, 'y_max': 150},  # (1,1)
    {'x_min': 100, 'x_max': 150, 'y_min': 100, 'y_max': 150},  # (1,3)
    {'x_min': 0, 'x_max': 50, 'y_min': 0, 'y_max': 50},  # (3,1)
    {'x_min': 100, 'x_max': 150, 'y_min': 0, 'y_max': 50}   # (3,3)
    ]

# Function to creaate a polytope for a given box

def box_to_polytope(bounds):
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Normal vectors
    b = np.array([-x_min, x_max, -y_min, y_max])      # Bounds
    return pt.Polytope(A, b)

# Create a polytope for each box
polytopes = [box_to_polytope(bounds) for bounds in obstacles]


# Define terminal set as a polytope
Af = np.array([[1, 0, 0],  # x <= 150
                       [-1, 0, 0],  # x >= 100
                       [0, 1, 0],  # y <= 100
                       [0, -1, 0]])  # y >= 50
bf = np.array([150, -100, 100, -50])

Xf = {'type': 'polytope', 'Af': Af, 'bf': bf}

z1,z2,z3,u1,u2=cftoc_tracking(z0,zf,N,Ts, Xf, polytopes)
"""
plt.figure(1)
plt.plot(z1,z2,'b')
plt.plot(z0[0],z0[1],'ro')
plt.plot(zf[0],zf[1],'bo')
plt.show()
"""


# plot obstacles
# function to plot a single polytope
def plot_polytope(polytope, ax, color='b', alpha=0.1):
    polytope.plot(ax, color=color, alpha=alpha, linestyle='solid', linewidth=1, edgecolor=None)

# Plot alll polytopes
fig, ax = plt.subplots(figsize=(8,8))    
for polytope in polytopes:e3w
    plot_polytope(polytope, ax)
# Plot reference trajectory on same plot as polytopes
#fig, ax = plt.subplots(figsize=(8,8))
for polytope in polytopes:
    plot_polytope(polytope, ax)
ax.plot(z1, z2, label="Trajectory", linestyle='--')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Reference Trajectory with Polytopes")
ax.grid(True)
ax.axis('equal')
plt.show()
