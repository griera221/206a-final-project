"""
12-10-24 By: George Riera

This code is being used to test the functionality of the polytope class.
"""

# Import packages
import numpy as np
import polytope as pt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# Define the vertices of the polytope
"""
A box can be represented as a polytope using a set of inequalities
A * x <= b

Where:
A: is the matrix of normal vecors to the faces of the box,
b defines the bounds.
"""


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

# function to plot a single polytope
def plot_polytope(polytope, ax, color='b', alpha=0.1):
    polytope.plot(ax, color=color, alpha=alpha, linestyle='solid', linewidth=1, edgecolor=None)

# Plot alll polytopes
fig, ax = plt.subplots(figsize=(8,8))    
for polytope in polytopes:
    plot_polytope(polytope, ax)

# Add labels and formating
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Multiple Polytopes Visualization")
ax.grid(True)
ax.axis('equal')
#plt.show()

# Reference Tracking Path

# Waypoints
waypoints_x = [75, 75, 75, 100, 125]
waypoints_y = [25, 50, 75, 75, 75]

# Time points
t = np.linspace(0, 1, len(waypoints_x))

# Interpolate
cs_x = CubicSpline(t, waypoints_x)
cs_y = CubicSpline(t, waypoints_y)

# Generate reference points
t_dense = np.linspace(0, 1, 100)  # Dense time array
x_ref = cs_x(t_dense)
y_ref = cs_y(t_dense)

# Plot reference trajectory on same plot as polytopes
fig, ax = plt.subplots(figsize=(8,8))
for polytope in polytopes:
    plot_polytope(polytope, ax)
ax.plot(x_ref, y_ref, label="Reference Trajectory", linestyle='--')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Reference Trajectory with Polytopes")
ax.grid(True)
ax.axis('equal')
plt.show()
