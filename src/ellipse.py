import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Kreis Parameter
radius = 5
circle_center = (0, 0)

# Gegebene Punkte innerhalb des Kreises
point1 = (-3, 0.1)
point2 = (-3, -2)
(x1, y1), (x2, y2) = point1, point2

# Funktion zur Berechnung der Ellipsenparameter
def ellipse_loss(params):
  b, angle = params
  a = radius
  cos_angle = np.cos(angle)
  sin_angle = np.sin(angle)

  # Punkt 1 auf der Ellipse
  term1 = ((x1 * cos_angle + y1 * sin_angle) ** 2) / a**2
  term2 = ((x1 * sin_angle - y1 * cos_angle) ** 2) / b**2

  # Punkt 2 auf der Ellipse
  term3 = ((x2 * cos_angle + y2 * sin_angle) ** 2) / a**2
  term4 = ((x2 * sin_angle - y2 * cos_angle) ** 2) / b**2

  return (term1 + term2 - 1) ** 2 + (term3 + term4 - 1) ** 2

# Initiale Schätzungen für b und angle
initial_guess = [radius / 2, 0]

# Optimiere die Parameter der Ellipse
result = minimize(ellipse_loss, initial_guess, method='Nelder-Mead')
b_opt, angle_opt = result.x
a_opt = radius

# Generiere Ellipse Punkte
t = np.linspace(0, 2 * np.pi, 100)
ellipse_x = a_opt * np.cos(t)
ellipse_y = b_opt * np.sin(t)

# Rotationsmatrix anwenden
R = np.array([[np.cos(angle_opt), -np.sin(angle_opt)],
              [np.sin(angle_opt), np.cos(angle_opt)]])

ellipse_rotated = np.dot(R, np.array([ellipse_x, ellipse_y]))

# Zeichne Kreis und Ellipse
fig, ax = plt.subplots()

# Kreis zeichnen
circle = plt.Circle(circle_center, radius, color='blue', fill=False)
ax.add_artist(circle)

# Ellipse zeichnen
ax.plot(ellipse_rotated[0, :] + circle_center[0], ellipse_rotated[1, :] + circle_center[1], color='red')

# Gegebene Punkte zeichnen
ax.plot(x1, y1, 'go')  # Punkt 1
ax.plot(x2, y2, 'go')  # Punkt 2

# Plot Konfiguration
ax.set_xlim(-radius - 1, radius + 1)
ax.set_ylim(-radius - 1, radius + 1)
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.title('Ellipse innerhalb eines Kreises')
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')

plt.show()
