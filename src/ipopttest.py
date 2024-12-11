from pyomo.environ import *
model = ConcreteModel()
model.x = Var(bounds=(0, 10), initialize=5)
model.obj = Objective(expr=(model.x - 2)**2)
SolverFactory('ipopt').solve(model, tee=True)
print("Optimal value of x:", model.x())