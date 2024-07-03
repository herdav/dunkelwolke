# orthodromeFromPolar
# Created 2024-07-02 by David Herren

import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tkinter as tk
from PIL import Image, ImageTk

# Constant for the diameter
DIAMETER = 1024
RADIUS = DIAMETER / 2

def create_great_circle_image(polar_coordinate1, polar_coordinate2):
  def polar_to_geographic(radius, angle):
    # Convert angle to radians
    angle_radians = math.radians(angle)

    # Calculate cartesian coordinates
    x = radius * math.cos(angle_radians)
    y = radius * math.sin(angle_radians)

    # Normalize coordinates to geographic scale
    latitude = (y / RADIUS) * 90
    longitude = (x / RADIUS) * 180
    
    return latitude, longitude
  
  def normalize_radius(radius):
    # Normalize the radius based on the diameter
    return (radius / 1024) * (DIAMETER / 2)
  
  # Normalize and convert the polar coordinates
  norm_radius1 = normalize_radius(polar_coordinate1[0])
  lat1, lon1 = polar_to_geographic(norm_radius1, polar_coordinate1[1])

  norm_radius2 = normalize_radius(polar_coordinate2[0])
  lat2, lon2 = polar_to_geographic(norm_radius2, polar_coordinate2[1])

  # Create a figure with a fixed size in inches (considering DPI)
  fig = plt.figure(figsize=(10.24, 10.24), dpi=150)

  # Create a Basemap instance with orthographic projection
  m = Basemap(projection='ortho', lat_0=0, lon_0=0)

  # Draw the great circle distance
  m.drawgreatcircle(lon1, lat1, lon2, lat2, linewidth=2, color='b')

  # Remove the Basemap background and only show the great circle line
  plt.gca().set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())

  # Convert plot to image
  canvas = FigureCanvas(fig)
  canvas.draw()
  image = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())
  plt.close(fig)
  return image

def show_image_in_window(image):
  # Create a Tkinter window
  window = tk.Tk()
  window.title("Great Circle Image")

  # Convert image to PhotoImage
  photo = ImageTk.PhotoImage(image)

  # Create a label to hold the image
  label = tk.Label(window, image=photo)
  label.pack()

  # Run the Tkinter main loop
  window.mainloop()

# Example usage
if __name__ == "__main__":
  polar_coordinate1 = (350, 10)  # (Radius, Angle)
  polar_coordinate2 = (450, 240)  # (Radius, Angle)
  image = create_great_circle_image(polar_coordinate1, polar_coordinate2)
  show_image_in_window(image)
