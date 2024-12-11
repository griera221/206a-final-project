import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt

#Scene parameters
d_bike      = 100  # m
w_bike  = 2    # m
l_car   = 4.5  #m
dsafe = d_bike - w_bike/2 - l_car/2

tmin   = 147   # s
tmax   = 150   # s

v_max   = 32      #m/s
g       = 9.81    #m/s^2
Ts      = 1       #s sampling time
nx = 2        # number of states
nu = 1         # number of inputs

def cftoc(x0, d_bike, N, mu, vref, Xf={'type':''},p={'type':''}):
  # x is state vector, u is input, Ts is sampling period.
  model = pyo.ConcreteModel()
  model.tidx = pyo.Set(initialize=range(0, N+1)) # length of finite optimization problem
  model.tidu = pyo.Set(initialize=range(0, N)) # length of finite optimization problem
  model.xidx = pyo.Set(initialize=range(0, nx))
  model.uidx = pyo.Set(initialize=range(0, nu))

  # Create state and input variables trajectory:
  model.x = pyo.Var(model.xidx, model.tidx)
  model.u = pyo.Var(model.uidx, model.tidu)

  # Tuning
  model.R=2
  model.Q=1

  # This code handles 3 types of terminal constraint: cvxhull, full-dimentional polytope, and terminal equality constraint
  if  Xf['type']=='cvxhull':
    model.nf = np.size(Xf['SS'], 0)
    model.nfidx = pyo.Set( initialize= range(model.nf), ordered=True )
    print( model.nf )
    model.SS = Xf['SS']
    model.lambdavar = pyo.Var(model.nfidx)

  if Xf['type']=='polytope':
    model.nf = np.size(Xf['Af'], 0)
    model.nfidx= pyo.Set( initialize= range(model.nf), ordered=True )
    model.Af = Xf['Af']
    model.bf = Xf['bf']

  if Xf['type']=='polytope_eq':
    model.nf = np.size(Xf['Aeq'], 0)
    model.nfidx= pyo.Set(initialize = range(model.nf), ordered=True )
    model.Aeq = Xf['Aeq']
    model.beq = Xf['beq']

  # This code handles 2 types of terminal cost: cvxhull, and quadratic cost
  if  p['type']=='cvxhull':
    model.cvalue=p['patsamples']

  if  p['type']=='quadratic':
    model.P=p['P']

  # Constraints:
  #Initial condition
  model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[i, 0] == x0[i])

  #State dynamics
  model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[0, t+1] == model.x[0, t] + Ts*model.x[1, t]
                                    if t < N else pyo.Constraint.Skip)
  model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t+1] == model.x[1, t] + Ts*model.u[0,t]
                                    if t < N else pyo.Constraint.Skip)

  #State bounds
  model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t] >= 0) # non-strict inequalities not allowed
  model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t] <= v_max) # Speed limit

  #Input bounds
  model.constraint6 = pyo.Constraint(model.tidu, rule=lambda model, t: model.u[0,t] >= -g*mu)
  model.constraint7 = pyo.Constraint(model.tidu, rule=lambda model, t: model.u[0,t] <= g*mu)

  #State constraints
  #model.constraint8 = pyo.Constraint(model.tidx, rule = lambda model, t: model.x[0,t] <= d_bike - w_bike/2-  l_car/2 )

  if  Xf['type']=='cvxhull':
    model.constraint9= pyo.Constraint(model.xidx, rule=lambda model, k: model.x[k,N] == sum( (model.SS[i,k] * model.lambdavar[i]) for i in model.nfidx))
    model.constraint10= pyo.Constraint(rule=lambda model: sum(model.lambdavar[i]  for i in model.nfidx)  == 1 )
    model.constraint11= pyo.Constraint(model.nfidx, rule=lambda model, i:   model.lambdavar[i]>=0)
  if  Xf['type']=='polytope':
    def final_const_rule(model, i):
        return sum(model.Af[i, j] * model.x[j, N] for j in model.xidx) <= model.bf[i]
    model.final_const = pyo.Constraint(model.nfidx, rule=final_const_rule)
  if  Xf['type']=='polytope_eq':
    def final_const_rule(model, i):
        return sum(model.Aeq[i, j] * model.x[j, N] for j in model.xidx) == model.beq[i]
    model.final_const = pyo.Constraint(model.nfidx, rule=final_const_rule)

  if p['type']=='cvxhull':
    #jerk minimization
    # model.cost = pyo.Objective(expr = sum((model.u[i, t+1]-model.u[i, t])**2 for i in model.uidx for t in model.tidx if t < N-1) + sum((model.x[1, t]-vref)**2 for t in model.tidx), sense=pyo.minimize)
    #acceleration minimization
    model.cost = pyo.Objective(expr = (sum(model.R*(model.u[0, t])**2  for t in model.tidu ) + sum(model.Q*(model.x[1, t]-vref)**2 for t in model.tidx) +sum((model.cvalue[i] * model.lambdavar[i]) for i in model.nfidx)), sense=pyo.minimize)
  elif  p['type']=='quadratic':
    model.cost = pyo.Objective(expr = (sum(model.R*(model.u[0, t])**2  for t in model.tidu ) + sum(model.Q*(model.x[1, t]-vref)**2 for t in model.tidu) +model.P*(model.x[1, N]-vref)**2), sense=pyo.minimize)
  else:
    model.cost = pyo.Objective(expr = (sum(model.R*(model.u[0, t])**2  for t in model.tidu ) + sum(model.Q*(model.x[1, t]-vref)**2 for t in model.tidu)), sense=pyo.minimize)

  # Now we can solve:
  results = pyo.SolverFactory('ipopt').solve(model)
  d = pyo.value(model.x[0,:])
  v = pyo.value(model.x[1,:])
  a = pyo.value(model.u[0,:])
  c = pyo.value(model.cost)

  return d, v, a, results.solver.termination_condition,c

##----------------------------

##----------------------------

### Initialization
x0    = np.array([40,10]) #[position, velocity]
T     = tmax #task horizon
vref  = 20 #reference speed
mu    = 0.5     #road friction coefficient
allow_reverse=False
N = T # prediction horizon
########

# strategy A
# terminal region constraint : Af*xN <= bf
Af = np.array([[1, 0]])
bf = np.zeros((1, )) + dsafe
Xf={'type':'polytope','Af':Af,'bf':bf}
p={'type':'quadratic','P':1}
d, v, a, termination_condition,c = cftoc(x0,d_bike,N,mu,vref,Xf,p)

# plot results
plt.figure(figsize=(8, 8), dpi=80)
plt.plot(d, v,'-og',label='Optimal solution over task T='+str(T)+', cost='+str(round(c)))
plt.axvline(x = dsafe, color = 'r', label = 'safety constraint')
plt.legend(fontsize=12)
plt.xlabel('d [m]', fontsize=24)
plt.ylabel('v [m/s]', fontsize=24)
plt.tick_params(axis='x', labelsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.show()

t_array = range(T+1)
f = plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.plot(t_array, d)
plt.plot(np.asarray(t_array), np.asarray(d)*0+ dsafe,'r--')
plt.ylabel('D (m)')
plt.subplot(3,1,2)
plt.plot(t_array, v)
plt.ylabel('v (m/s)')
plt.subplot(3,1,3)
plt.plot(t_array[:-1], a,'o')
plt.ylabel('a ($m/s^2$)')
plt.xlabel('t (s)')
plt.show()