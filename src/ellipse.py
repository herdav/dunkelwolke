import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Circle parameters
radius = 5
circle_center = (0, 0)

# Given points within the circle
point1 = (1, 0.5)  # Example point 1
point2 = (-3, -2)  # Example point 2
(x1, y1), (x2, y2) = point1, point2

# Function to calculate ellipse parameters
def ellipse_loss(params):
  b, angle = params
  a = radius
  cos_angle = np.cos(angle)
  sin_angle = np.sin(angle)

  # Point 1 on the ellipse
  term1 = ((x1 * cos_angle + y1 * sin_angle) ** 2) / a**2
  term2 = ((x1 * sin_angle - y1 * cos_angle) ** 2) / b**2

  # Point 2 on the ellipse
  term3 = ((x2 * cos_angle + y2 * sin_angle) ** 2) / a**2
  term4 = ((x2 * sin_angle - y2 * cos_angle) ** 2) / b**2

  return (term1 + term2 - 1) ** 2 + (term3 + term4 - 1) ** 2

# Initial guesses for b and angle
initial_guess = [radius / 2, 0]

# Optimize the ellipse parameters
result = minimize(ellipse_loss, initial_guess, method='Nelder-Mead')
b_opt, angle_opt = result.x
a_opt = radius

# Ensure points are on the same side of the ellipse axis
def points_on_same_side(angle, x1, y1, x2, y2):
  cos_angle = np.cos(angle)
  sin_angle = np.sin(angle)
  return ((x1 * cos_angle + y1 * sin_angle) * (x2 * cos_angle + y2 * sin_angle)) >= 0

if not points_on_same_side(angle_opt, x1, y1, x2, y2):
  angle_opt += np.pi  # Adjust angle by 180 degrees if points are on opposite sides

# Generate ellipse points
t = np.linspace(0, 2 * np.pi, 1000)
ellipse_x = a_opt * np.cos(t)
ellipse_y = b_opt * np.sin(t)

# Apply rotation matrix
R = np.array([[np.cos(angle_opt), -np.sin(angle_opt)],
              [np.sin(angle_opt), np.cos(angle_opt)]])

ellipse_rotated = np.dot(R, np.array([ellipse_x, ellipse_y]))

# Plot circle and ellipse
fig, ax = plt.subplots()

# Draw circle
circle = plt.Circle(circle_center, radius, color='blue', fill=False)
ax.add_artist(circle)

# Draw ellipse
ax.plot(ellipse_rotated[0, :] + circle_center[0], ellipse_rotated[1, :] + circle_center[1], color='red')

# Plot given points
ax.plot(x1, y1, 'go')  # Point 1
ax.plot(x2, y2, 'go')  # Point 2

# Plot configuration
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.title('Ellipse within a Circle')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.show()
