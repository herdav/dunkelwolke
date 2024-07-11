# kMeansVideo
# Created 2024-07-11 by David Herren

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import mark_boundaries
import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from mpl_toolkits.basemap import Basemap
import math
from geopy.distance import geodesic
from geopy.point import Point

class VideoClusterFilterApp:
  # Colors for visualization
  BOUNDARY_COLOR = (1, 1, 1)
  
  CENTER_ORG_COLOR = (0, 0, 255)
  CIRCLE_ORG_COLOR = (0, 0, 255)
  NUMBER_ORG_COLOR = (0, 0, 255)
  
  CENTER_VAL_COLOR = (0, 255, 0)
  CIRCLE_VAL_COLOR = (0, 255, 0)
  NUMBER_VAL_COLOR = (0, 255, 0)
  
  CONNEC_VAL_COLOR = (255, 255, 0)
  PATH_VAL_COLOR = (0, 255, 0)
  CONNEC_LENGTH_COLOR = (255, 255, 0)

  # Colors for export
  EXPORT_CENTER_ORG_COLOR = (255, 255, 255)
  EXPORT_CIRCLE_ORG_COLOR = (255, 255, 255)
  EXPORT_NUMBER_ORG_COLOR = (255, 255, 255)
  
  EXPORT_CENTER_VAL_COLOR = (255, 255, 255)
  EXPORT_CIRCLE_VAL_COLOR = (255, 255, 255)
  EXPORT_NUMBER_VAL_COLOR = (255, 255, 255)

  EXPORT_CONNEC_VAL_COLOR = (255, 255, 255)
  EXPORT_PATH_VAL_COLOR = (255, 255, 255)
  EXPORT_CONNEC_LENGTH_COLOR = (255, 255, 255)

  # Thickness for visualization
  CENTER_ORG_THICKNESS = 1
  CIRCLE_ORG_THICKNESS = 1
  
  CENTER_VAL_THICKNESS = 1
  CIRCLE_VAL_THICKNESS = 1
  
  CONNEC_VAL_THICKNESS = 1
  PATH_VAL_THICKNESS = 1
  
  # Thickness for export
  EXPORT_CENTER_ORG_THICKNESS = 1
  EXPORT_CIRCLE_ORG_THICKNESS = 1

  EXPORT_CENTER_VAL_THICKNESS = 1
  EXPORT_CIRCLE_VAL_THICKNESS = 1
  
  EXPORT_CONNEC_VAL_THICKNESS = 1
  EXPORT_PATH_VAL_THICKNESS = 1

  # Font size for numbers
  FONT_SIZE = 0.4
  FONT_THICKNESS = 1
  
  # Parameters
  NUM_COLORS = 4
  SELECTED_CLUSTER = 0
  GRAYSCALE_VAR = 1
  RADIUS_VAR = 80
  FILL_CLUSTER_VAR = 0
  SECOND_KMEANS_CLUSTERS_VAR = 4
  MERGE_THRESHOLD_VAR = 20
  T_VAL_VAR = 10
  T_EXIST_VAR = 20
  CONNECTION_COUNT_VAR = 1
  EXPORT_SEPARATE_VAR = 1
  SHOW_CLUSTER_NUMBER_VAR = 1
  SHOW_SECOND_KMEANS_VAR = 1
  SHOW_ALL_PATHS_VAR = 0
  SHOW_CONNECTION_LENGTH_VAR = 0
  
  def __init__(self, root):
    self.root = root
    self.root.title("k-Means Video Cluster Filter")

    # Initial parameters for k-means
    self.num_colors = self.NUM_COLORS
    self.selected_cluster = self.SELECTED_CLUSTER
    self.preview_frame = None
    self.current_frame_index = 0
    self.total_frames = 0
    self.fps = 18

    # Initialize the K-means algorithm globally
    self.kmeans = None

    # Variables for cluster centers and frames
    self.center_org = []
    self.center_val = []
    self.t_val_frames = []
    self.t_org_frames = []
    self.center_paths = []
    #self.all_paths = []  # To store all paths ever drawn

    # Stop processing flag
    self.stop_processing = False

    self.create_gui()

  def create_gui(self):
    left_frame = tk.Frame(self.root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)
    self.preview_label = tk.Label(left_frame)
    self.preview_label.pack()

    right_frame = tk.Frame(self.root)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.BOTH, expand=True)

    btn_select_video = tk.Button(right_frame, text="Select Video", command=self.select_video)
    btn_select_video.grid(row=0, column=0, pady=5)

    btn_reset = tk.Button(right_frame, text="Reset", command=self.reset_parameters)
    btn_reset.grid(row=0, column=1, pady=5)

    self.video_path_label = tk.Label(right_frame, text="No Video is Loaded!")
    self.video_path_label.grid(row=0, column=3, columnspan=2, pady=2)

    tk.Label(right_frame, text="Clusters").grid(row=2, column=0, pady=2)
    self.num_colors_label = tk.Label(right_frame, text=str(self.NUM_COLORS))
    self.num_colors_label.grid(row=2, column=1, pady=2)

    btn_increase_colors = tk.Button(right_frame, text="+", command=self.increase_colors)
    btn_increase_colors.grid(row=3, column=0, pady=2)
    btn_decrease_colors = tk.Button(right_frame, text="-", command=self.decrease_colors)
    btn_decrease_colors.grid(row=3, column=1, pady=2)

    tk.Label(right_frame, text="Boundary").grid(row=4, column=0, pady=2)
    self.selected_cluster_label = tk.Label(right_frame, text=str(self.SELECTED_CLUSTER))
    self.selected_cluster_label.grid(row=4, column=1, pady=2)

    btn_increase_cluster = tk.Button(right_frame, text="+", command=self.increase_cluster)
    btn_increase_cluster.grid(row=5, column=0, pady=2)
    btn_decrease_cluster = tk.Button(right_frame, text="-", command=self.decrease_cluster)
    btn_decrease_cluster.grid(row=5, column=1, pady=2)

    tk.Label(right_frame, text="Radius (%)").grid(row=7, column=0, pady=2, sticky='w')
    self.radius_var = tk.IntVar(value=self.RADIUS_VAR)
    tk.Scale(right_frame, from_=0, to=100, orient='horizontal', variable=self.radius_var, command=self.update_preview).grid(row=7, column=1, pady=2, sticky='w')

    tk.Label(right_frame, text="Second K-means Clusters").grid(row=8, column=0, pady=2, sticky='w')
    self.second_kmeans_clusters_var = tk.IntVar(value=self.SECOND_KMEANS_CLUSTERS_VAR)
    tk.Scale(right_frame, from_=1, to=20, orient='horizontal', variable=self.second_kmeans_clusters_var, command=self.update_preview).grid(row=8, column=1, pady=2, sticky='w')

    tk.Label(right_frame, text="Merge Threshold (%)").grid(row=9, column=0, pady=2, sticky='w')
    self.merge_threshold_var = tk.IntVar(value=self.MERGE_THRESHOLD_VAR)
    tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=self.merge_threshold_var, command=self.update_preview).grid(row=9, column=1, pady=2, sticky='w')

    tk.Label(right_frame, text="t_val (frames)").grid(row=10, column=0, pady=2, sticky='w')
    self.t_val_var = tk.IntVar(value=self.T_VAL_VAR)
    tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=self.t_val_var, command=self.update_preview).grid(row=10, column=1, pady=2, sticky='w')

    tk.Label(right_frame, text="t_exist (frames)").grid(row=11, column=0, pady=2, sticky='w')
    self.t_exist_var = tk.IntVar(value=self.T_EXIST_VAR)
    tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=self.t_exist_var, command=self.update_preview).grid(row=11, column=1, pady=2, sticky='w')

    tk.Label(right_frame, text="Number of Connections").grid(row=12, column=0, pady=2, sticky='w')
    self.connection_count_var = tk.IntVar(value=self.CONNECTION_COUNT_VAR)
    tk.Scale(right_frame, from_=0, to=4, orient='horizontal', variable=self.connection_count_var, command=self.update_preview).grid(row=12, column=1, pady=2, sticky='w')

    self.export_original_var = tk.IntVar(value=1)
    self.export_separate_var = tk.IntVar(value=self.EXPORT_SEPARATE_VAR)
    self.grayscale_var = tk.IntVar(value=self.GRAYSCALE_VAR)
    self.fill_cluster_var = tk.IntVar(value=self.FILL_CLUSTER_VAR)
    self.show_second_kmeans_var = tk.IntVar(value=self.SHOW_SECOND_KMEANS_VAR)
    self.center_val_var = tk.IntVar(value=1)
    self.show_cluster_number_var = tk.IntVar(value=self.SHOW_CLUSTER_NUMBER_VAR)
    self.show_all_paths_var = tk.IntVar(value=self.SHOW_ALL_PATHS_VAR)
    self.show_connection_length_var = tk.IntVar(value=self.SHOW_CONNECTION_LENGTH_VAR)

    chk_export_original = tk.Checkbutton(right_frame, text="Export Original Video", variable=self.export_original_var)
    chk_export_original.grid(row=13, column=0, sticky='w', pady=2)

    chk_export_separate = tk.Checkbutton(right_frame, text="Export All Layers Separately", variable=self.export_separate_var)
    chk_export_separate.grid(row=14, column=0, sticky='w', pady=2)

    chk_grayscale = tk.Checkbutton(right_frame, text="Display Grayscale Video", variable=self.grayscale_var, command=self.update_preview)
    chk_grayscale.grid(row=15, column=0, sticky='w', pady=2)

    chk_fill_cluster = tk.Checkbutton(right_frame, text="Fill Selected Cluster", variable=self.fill_cluster_var, command=self.update_preview)
    chk_fill_cluster.grid(row=16, column=0, sticky='w', pady=2)

    chk_show_second_kmeans = tk.Checkbutton(right_frame, text="Show Second K-means", variable=self.show_second_kmeans_var, command=self.update_preview)
    chk_show_second_kmeans.grid(row=17, column=0, sticky='w', pady=2)

    chk_center_val = tk.Checkbutton(right_frame, text="Show Center Val", variable=self.center_val_var, command=self.update_preview)
    chk_center_val.grid(row=18, column=0, sticky='w', pady=2)

    chk_show_cluster_number = tk.Checkbutton(right_frame, text="Show Cluster Number", variable=self.show_cluster_number_var, command=self.update_preview)
    chk_show_cluster_number.grid(row=19, column=0, sticky='w', pady=2)

    chk_show_all_paths = tk.Checkbutton(right_frame, text="Show All Paths", variable=self.show_all_paths_var, command=self.update_preview)
    chk_show_all_paths.grid(row=20, column=0, sticky='w', pady=2)

    chk_show_connection_length = tk.Checkbutton(right_frame, text="Show Connection Length", variable=self.show_connection_length_var, command=self.update_preview)
    chk_show_connection_length.grid(row=21, column=0, sticky='w', pady=2)

    self.frame_slider = tk.Scale(right_frame, from_=0, to=self.total_frames - 1, orient='horizontal', command=self.update_frame)
    self.frame_slider.grid(row=22, column=0, columnspan=4, pady=2, sticky="ew")

    self.progress_bar = ttk.Progressbar(right_frame, orient='horizontal', mode='determinate')
    self.progress_bar.grid(row=23, column=0, columnspan=2, pady=20, sticky="ew")
    self.progress_label = tk.Label(right_frame, text="Frame 0/0")
    self.progress_label.grid(row=23, column=2, columnspan=2, pady=2, sticky="ew")

    btn_start_processing = tk.Button(right_frame, text="Export Video", command=self.start_processing)
    btn_start_processing.grid(row=24, column=0, pady=2, sticky='ew')

    btn_stop_processing = tk.Button(right_frame, text="Stop", command=self.stop_processing)
    btn_stop_processing.grid(row=24, column=1, pady=2, sticky='ew')

    btn_exit = tk.Button(right_frame, text="Exit", command=self.root.destroy)
    btn_exit.grid(row=25, column=3, pady=2, sticky='ew')

  def get_unique_filename(self, base_path, base_name, ext):
    id = 0
    while os.path.exists(f"{base_path}/{base_name}_{id}{ext}"):
      id += 1
    return f"{base_path}/{base_name}_{id}{ext}"

  def process_frame(self, frame, num_colors, selected_cluster, boundary_color, radius, kmeans=None, grayscale=False, fill_cluster=False):
    if grayscale:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255), thickness=-1)

    image = cv2.bitwise_and(frame, frame, mask=mask)
    image_np = np.array(image)

    if image_np.shape[2] == 4:
      pixels = image_np.reshape(-1, 4)  # RGBA image
    else:
      pixels = image_np.reshape(-1, 3)  # RGB image

    if kmeans is None:
      kmeans = MiniBatchKMeans(n_clusters=num_colors, batch_size=1000, random_state=42)
      kmeans.fit(pixels)
    else:
      kmeans.n_clusters = num_colors
      kmeans.partial_fit(pixels)

    labels = kmeans.predict(pixels)
    centroids = kmeans.cluster_centers_

    segmented_img = centroids[labels].reshape(image_np.shape).astype(np.uint8)
    mask_labels = (labels == selected_cluster)
    masked_labels = np.where(mask_labels.reshape(image_np.shape[:2]), labels.reshape(image_np.shape[:2]), -1)

    if fill_cluster:
      new_fill_area = mask.copy()
      new_fill_area[mask_labels.reshape(image_np.shape[:2])] = 0
      fill_img = np.zeros_like(frame)
      fill_img[np.where(new_fill_area == 255)] = [int(c * 255) for c in boundary_color]
      inverted_fill_img = np.zeros_like(frame)
      inverted_fill_img[np.where(mask_labels.reshape(image_np.shape[:2]))] = [int(c * 255) for c in boundary_color]
      inverted_fill_img = cv2.bitwise_and(inverted_fill_img, inverted_fill_img, mask=mask)
      frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
      frame = cv2.add(frame, fill_img)
    else:
      fill_img = None
      inverted_fill_img = None

    boundary_img = mark_boundaries(np.zeros_like(segmented_img), masked_labels, color=boundary_color, mode='thick')
    boundary_img = (boundary_img * 255).astype(np.uint8)
    frame_with_boundaries = cv2.addWeighted(frame, 1, boundary_img, 1, 0)
    cv2.circle(frame_with_boundaries, center, radius, (255, 255, 255), 2)

    return frame_with_boundaries, boundary_img, kmeans, inverted_fill_img

  def update_preview(self, val=None):
    if self.preview_frame is not None:
      radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2))
      preview_with_boundaries, _, self.kmeans, filled_only_img = self.process_frame(
        self.preview_frame, self.num_colors, self.selected_cluster, self.BOUNDARY_COLOR, radius,
        self.kmeans, self.grayscale_var.get(), self.fill_cluster_var.get()
      )

      if filled_only_img is None:
        _, _, _, filled_only_img = self.process_frame(
          self.preview_frame, self.num_colors, self.selected_cluster, self.BOUNDARY_COLOR, radius,
          self.kmeans, self.grayscale_var.get(), True
        )

      second_kmeans_clusters = self.second_kmeans_clusters_var.get()
      second_centers = self.perform_second_kmeans(filled_only_img, second_kmeans_clusters)
      second_centers = self.merge_close_centers(second_centers, self.merge_threshold_var.get(), self.preview_frame.shape[:2])
      self.update_centers(second_centers)

      if self.show_second_kmeans_var.get():
        self.draw_lines_and_markers(preview_with_boundaries, second_centers, self.CONNEC_VAL_COLOR, self.CONNEC_VAL_THICKNESS)

      if self.center_val_var.get():
        self.draw_center_val(preview_with_boundaries, self.CENTER_VAL_COLOR, self.CENTER_VAL_THICKNESS)
        self.draw_center_paths(preview_with_boundaries, self.PATH_VAL_COLOR, self.PATH_VAL_THICKNESS)

      if self.show_all_paths_var.get():
        self.draw_all_paths(preview_with_boundaries, self.PATH_VAL_COLOR, self.PATH_VAL_THICKNESS)

      self.draw_center_numbers(preview_with_boundaries, self.NUMBER_ORG_COLOR, self.FONT_SIZE, self.CENTER_ORG_THICKNESS, self.center_org)

      preview_image = Image.fromarray(cv2.cvtColor(preview_with_boundaries, cv2.COLOR_BGR2RGB))
      preview_photo = ImageTk.PhotoImage(preview_image)
      self.preview_label.config(image=preview_photo)
      self.preview_label.image = preview_photo

      self.num_colors_label.config(text=str(self.num_colors))
      self.selected_cluster_label.config(text=str(self.selected_cluster))

  def image_to_cartesian(self, x, y):
    height, width = self.preview_frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    cartesian_x = x - center_x 
    cartesian_y = center_y - y
    return cartesian_x, cartesian_y

  def carToGeo(self, x, z, r): # z is y in image
    y_squared = r**2 - x**2 - z**2
    y = math.sqrt(y_squared)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    lat = (math.degrees(theta) - 90) * -1
    lon = (math.degrees(phi) - 90) * -1
    
    return lat, lon

  def cartesian_to_geographic(self, x, y, image):
      cartesian_x, cartesian_y = self.image_to_cartesian(x, y)
      radius = (int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2)))
      latitude, longitude = self.carToGeo(cartesian_x, cartesian_y, radius)
      return latitude, longitude

  def draw_great_circle_only(self, image, color, thickness):
      radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2))
      for i, center in enumerate(self.center_val):
        distances = np.linalg.norm(np.array(self.center_val) - center, axis=1)
        closest_indices = distances.argsort()[1:self.connection_count_var.get() + 1]
        
        for idx in closest_indices:
          closest_center = self.center_val[idx]
          lon1 = self.cartesian_to_geographic(int(center[1]), int(center[0]), image)[1]
          lat1 = self.cartesian_to_geographic(int(center[1]), int(center[0]), image)[0]
          lon2 = self.cartesian_to_geographic(int(closest_center[1]), int(closest_center[0]), image)[1]
          lat2 = self.cartesian_to_geographic(int(closest_center[1]), int(closest_center[0]), image)[0]
          
          great_circle = GreatCircleSegments(lon1, lat1, lon2, lat2)
          center_x = image.shape[1] // 2
          center_y = image.shape[0] // 2
          
          radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1]))

          image = great_circle.draw_great_circle(image, center_x, center_y, radius, color)

  def draw_center_ellipse_only(self, image, color, thikness):
    for i, center in enumerate(self.center_val):
      angle, r = self.calculate_angle_and_distance(int(center[1]), int(center[0]), 512, 512)
      ellipse_plotter = CenterEllipse(int(center[1]), int(center[0]), r, angle, 819.2, self.get_merge_threshold(image.shape)/2, color, thikness)
      ellipse_plotter.draw_ellipse(image)

  def draw_geographic_coords(self, image, center, color, index=None):
    geographic_coords = self.cartesian_to_geographic(int(center[1]), int(center[0]), image)
    if index is not None:
      center_string = f"[{index}] {geographic_coords[0]:.1f};{geographic_coords[1]:.1f}"
    else:
      center_string = f"[{geographic_coords[0]:.1f};{geographic_coords[1]:.1f}"
      
    cv2.putText(image, center_string, 
          (int(center[1]) + self.get_merge_threshold(image.shape) + 10, int(center[0])), 
          cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, color, self.FONT_THICKNESS)

  def calculate_angle_and_distance(self, x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle = math.degrees(math.atan2(delta_y, delta_x))
    distance = math.sqrt(delta_x**2 + delta_y**2)
    return angle, distance

  def draw_lines_and_markers(self, image, centers, color, thickness):
    radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2))
    for i, center in enumerate(self.center_val):
      distances = np.linalg.norm(np.array(self.center_val) - center, axis=1)
      closest_indices = distances.argsort()[1:self.connection_count_var.get() + 1]
      for idx in closest_indices:
        closest_center = self.center_val[idx]
        cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), (255, 0, 0), thickness)

        lon1 = self.cartesian_to_geographic(int(center[1]), int(center[0]), image)[1]
        lat1 = self.cartesian_to_geographic(int(center[1]), int(center[0]), image)[0]
        lon2 = self.cartesian_to_geographic(int(closest_center[1]), int(closest_center[0]), image)[1]
        lat2 = self.cartesian_to_geographic(int(closest_center[1]), int(closest_center[0]), image)[0]

        great_circle = GreatCircleSegments(lon1, lat1, lon2, lat2)
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1]))
        image = great_circle.draw_great_circle(image, center_x, center_y, radius, (0, 0, 255))
        
        cv2.circle(image, (int(center[1]), int(center[0])), self.get_merge_threshold(image.shape), self.CIRCLE_VAL_COLOR, self.CIRCLE_VAL_THICKNESS)
        
        angle, r = self.calculate_angle_and_distance(int(center[1]), int(center[0]), 512, 512)
        ellipse_plotter = CenterEllipse(int(center[1]), int(center[0]), r, angle, 819.2, self.get_merge_threshold(image.shape)/2, self.CIRCLE_VAL_COLOR, self.CIRCLE_VAL_THICKNESS)
        ellipse_plotter.draw_ellipse(image)

        self.draw_geographic_coords(image, center, (0, 0, 255), index=i)

        if self.show_connection_length_var.get():
          mid_point = (int((center[1] + closest_center[1]) // 2), int((center[0] + closest_center[0]) // 2))
          length = np.linalg.norm(np.array(center) - np.array(closest_center))
          hav_length = self.haversine_distance(center[:2], closest_center[:2])
          cv2.putText(image, f"{hav_length / radius:.3f} ({length / radius:.3f})", (mid_point[0], mid_point[1]),  
                      cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.CONNEC_LENGTH_COLOR, self.FONT_THICKNESS)

  def draw_center_numbers(self, image, color, font_size, thickness, centers):
    for i, center in enumerate(centers):
      self.draw_geographic_coords(image, center, color, index=i)
      
  def cartesian_to_polar(self, x, y):
    center_x, center_y = self.preview_frame.shape[1] // 2, self.preview_frame.shape[0] // 2
    dx = x - center_x
    dy = center_y - y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta)
    if theta_deg < 0:
      theta_deg += 360
    return r, theta_deg
  
  def haversine_distance(self, coord1, coord2):
    # Convert polar coordinates to cartesian coordinates
    def polar_to_cartesian(r, theta_deg):
      theta_rad = np.radians(theta_deg)
      x = r * np.cos(theta_rad)
      y = r * np.sin(theta_rad)
      return x, y

    r1, theta1 = self.cartesian_to_polar(coord1[1], coord1[0])
    r2, theta2 = self.cartesian_to_polar(coord2[1], coord2[0])

    R = self.radius_var.get() / 100 * (self.preview_frame.shape[1] / 2)  # Radius corresponds to 80% of the image width
    if R == 0:  # In case RADIUS_VAR is set to 0
      R = 409.6

    # Get cartesian coordinates
    x1, y1 = polar_to_cartesian(r1, theta1)
    x2, y2 = polar_to_cartesian(r2, theta2)
    
    # Calculate cartesian distance
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx*dx + dy*dy)
    
    # Haversine formula to calculate the distance
    haversine_dist = 2 * R * np.arcsin(d / (2 * R))
    return haversine_dist

  def get_merge_threshold(self, image_shape):
    return int(self.merge_threshold_var.get() / 100 * (min(image_shape[:2]) / 2))

  def draw_center_val(self, image, color, thickness, draw_text=False, draw_connections=False, draw_circles=False):
    radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2))
    for i, center in enumerate(self.center_val):
      cv2.drawMarker(image, (int(center[1]), int(center[0])), color=color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=thickness)
      r, theta_deg = self.cartesian_to_polar(center[1], center[0])
      if draw_circles:
        cv2.circle(image, (int(center[1]), int(center[0])), self.get_merge_threshold(image.shape), self.CIRCLE_VAL_COLOR, self.CIRCLE_VAL_THICKNESS)
      if draw_text and self.show_cluster_number_var.get():
        polar_coord_str = f"({r/radius:.3f}:{int(theta_deg)})"
        cv2.putText(image, f"{i} {polar_coord_str}", 
                    (int(center[1]) + self.get_merge_threshold(image.shape) + 10, int(center[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.NUMBER_VAL_COLOR, self.FONT_THICKNESS)
      if draw_connections:
        distances = np.linalg.norm(np.array(self.center_val) - center, axis=1)
        closest_indices = distances.argsort()[1:self.connection_count_var.get()+1]
        for idx in closest_indices:
          closest_center = self.center_val[idx]
          cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), self.CONNEC_VAL_COLOR, self.CONNEC_VAL_THICKNESS)

  def draw_center_org(self, image, color, thickness):
    for i, center in enumerate(self.center_org):
      cv2.drawMarker(image, (int(center[1]), int(center[0])), color=color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=thickness)
      cv2.circle(image, (int(center[1]), int(center[0])), self.get_merge_threshold(image.shape), self.CIRCLE_ORG_COLOR, self.CIRCLE_ORG_THICKNESS)
      if self.show_cluster_number_var.get():
        cv2.putText(image, str(i), (int(center[1]) + self.get_merge_threshold(image.shape) + 10, int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.NUMBER_ORG_COLOR, self.FONT_THICKNESS)

  def draw_center_paths(self, image, color, thickness):
    for path in self.center_paths:
      if len(path) > 1:
        for j in range(1, len(path)):
          cv2.line(image, (int(path[j-1][1]), int(path[j-1][0])), (int(path[j][1]), int(path[j][0])), color, thickness)

  def draw_all_paths(self, image, color, thickness):
    for path in self.all_paths:
      if len(path) > 1:
       for j in range(1, len(path)):
         cv2.line(image, (int(path[j-1][1]), int(path[j-1][0])), (int(path[j][1]), int(path[j][0])), color, thickness)

  def draw_center_markers(self, image, centers, color, thickness):
    for center in centers:
      cv2.drawMarker(image, (int(center[1]), int(center[0])), color=color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=thickness)
      for center in self.center_val:
        cv2.drawMarker(image, (int(center[1]), int(center[0])), color=color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=thickness)

  def draw_connections_only(self, image, color, thickness):  
    for i, center in enumerate(self.center_val):
      distances = np.linalg.norm(np.array(self.center_val) - center, axis=1)
      closest_indices = distances.argsort()[1:self.connection_count_var.get()+1]
      for idx in closest_indices:
        closest_center = self.center_val[idx]
        cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), color, thickness)

  def draw_center_circles(self, image, color, thickness):
    for center in self.center_val:
      cv2.circle(image, (int(center[1]), int(center[0])), self.get_merge_threshold(image.shape), color, thickness)

  def draw_center_org_circles(self, image, color, thickness):
    for center in self.center_org:
      cv2.circle(image, (int(center[1]), int(center[0])), self.get_merge_threshold(image.shape), color, thickness)

  def decrease_colors(self):
    self.num_colors = max(1, self.num_colors - 1)
    if self.selected_cluster >= self.num_colors:
      self.selected_cluster = self.num_colors - 1
    self.kmeans = None
    self.update_preview()

  def increase_colors(self):
    self.num_colors += 1
    self.kmeans = None
    self.update_preview()

  def decrease_cluster(self):
    self.selected_cluster = max(0, self.selected_cluster - 1)
    self.update_preview()

  def increase_cluster(self):
    if self.selected_cluster < self.num_colors - 1:
      self.selected_cluster += 1
      self.update_preview()

  def toggle_grayscale(self):
    self.update_preview()

  def perform_second_kmeans(self, image, n_clusters):
    height, width, _ = image.shape
    Y, X = np.ogrid[:height, :width]
    mask = np.any(image != [0, 0, 0], axis=-1)  # Only use filled pixels
    positions = np.column_stack(np.where(mask))

    if positions.size == 0:
      return []

    data = image[mask].reshape(-1, 3)
    if data.size == 0:
      return []

    data = np.hstack((positions, data))

    # Ensure n_clusters is not greater than the number of samples
    n_clusters = min(n_clusters, len(data))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_
    return centers

  def merge_close_centers(self, centers, threshold_percent, image_shape):
    threshold = (threshold_percent / 100) * (min(image_shape[:2]) / 2)
    merged_centers = []
    while len(centers) > 0:
      center = centers[0]
      distances = np.linalg.norm(centers - center, axis=1)
      close_centers = centers[distances < threshold]
      centers = centers[distances >= threshold]
      merged_center = np.mean(close_centers, axis=0)
      merged_centers.append(merged_center)
    return np.array(merged_centers)

  def update_centers(self, new_centers):
    merge_threshold = self.merge_threshold_var.get()
    t_val = self.t_val_var.get()
    t_exist = self.t_exist_var.get()
    max_centers = self.second_kmeans_clusters_var.get()

    if len(self.center_val) == 0:
      self.center_val = new_centers.tolist()[:max_centers]
      self.t_val_frames = [t_val] * len(self.center_val)
      self.center_paths = [[center] for center in self.center_val]
      #self.all_paths.extend(self.center_paths)  # Add initial paths to all_paths
      return

    for new_center in new_centers:
      merged = False
      for i, val_center in enumerate(self.center_val):
        if np.linalg.norm(new_center[:2] - val_center[:2]) < merge_threshold:
          self.center_val[i] = (np.array(val_center) * (self.t_val_frames[i] - 1) + np.array(new_center)) / self.t_val_frames[i]
          self.t_val_frames[i] = t_val
          self.center_paths[i].append(new_center)
          merged = True
          break
      if not merged:
        self.center_org.append(new_center.tolist())
        self.t_org_frames.append(t_exist)
        #self.all_paths.append([new_center])  # Add new paths to all_paths

    self.merge_existing_centers(merge_threshold)
    self.remove_expired_centers(merge_threshold, t_val)
    self.trim_extra_centers(max_centers)

  def merge_existing_centers(self, merge_threshold):
    i = 0
    while i < len(self.center_val):
      j = i + 1
      while j < len(self.center_val):
        if np.linalg.norm(np.array(self.center_val[i][:2]) - np.array(self.center_val[j][:2])) < merge_threshold:
          self.center_val[i] = (np.array(self.center_val[i]) + np.array(self.center_val[j])) / 2
          self.center_paths[i].extend(self.center_paths[j])
          del self.center_val[j]
          del self.t_val_frames[j]
          del self.center_paths[j]
        else:
          j += 1
      i += 1

    for i in range(len(self.center_val)):
      if self.t_val_frames[i] > 1:
        self.center_val[i] = (np.array(self.center_val[i]) * (self.t_val_frames[i] - 1) + np.array(self.center_val[i])) / self.t_val_frames[i]
      self.t_val_frames[i] -= 1

    indices_to_keep = [i for i in range(len(self.center_val)) if self.t_val_frames[i] > 0]
    self.center_val = [self.center_val[i] for i in indices_to_keep]
    self.t_val_frames = [self.t_val_frames[i] for i in indices_to_keep]
    self.center_paths = [self.center_paths[i] for i in indices_to_keep]

  def remove_expired_centers(self, merge_threshold, t_val):
    new_stable_centers = []
    new_t_val_frames = []
    new_paths = []
    for i in range(len(self.center_org)):
      self.t_org_frames[i] -= 1
      if self.t_org_frames[i] <= 0:
        existing = False
        for j, val_center in enumerate(self.center_val):
          if np.linalg.norm(np.array(self.center_org[i][:2]) - np.array(val_center[:2])) < merge_threshold:
            existing = True
            break
        if not existing:
          new_stable_centers.append(self.center_org[i])
          new_t_val_frames.append(t_val)
          new_paths.append([self.center_org[i]])

    self.center_val.extend(new_stable_centers)
    self.t_val_frames.extend(new_t_val_frames)
    self.center_paths.extend(new_paths)
    indices_to_keep = [i for i in range(len(self.center_org)) if self.t_org_frames[i] > 0]
    self.center_org = [self.center_org[i] for i in indices_to_keep]
    self.t_org_frames = [self.t_org_frames[i] for i in indices_to_keep]

    i = 0
    while i < len(self.center_val):
      j = i + 1
      while j < len(self.center_val):
        if np.linalg.norm(np.array(self.center_val[i][:2]) - np.array(self.center_val[j][:2])) < merge_threshold:
          self.center_val[i] = (np.array(self.center_val[i]) + np.array(self.center_val[j])) / 2
          self.center_paths[i].extend(self.center_paths[j])
          del self.center_val[j]
          del self.t_val_frames[j]
          del self.center_paths[j]
        else:
          j += 1
      i += 1

  def trim_extra_centers(self, max_centers):
    if len(self.center_val) > max_centers:
      self.center_val = self.center_val[:max_centers]
      self.t_val_frames = self.t_val_frames[:max_centers]
      self.center_paths = self.center_paths[:max_centers]

  def start_processing(self):
    self.stop_processing = False
    export_original = self.export_original_var.get()
    export_separate = self.export_separate_var.get()
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if output_dir:
      self.process_video(export_original, export_separate, output_dir)

  def stop_processing(self):
    self.stop_processing = True

  def process_video(self, export_original, export_separate, output_dir):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if export_original:
      output_path = self.get_unique_filename(output_dir, 'output', '.avi')
      out = cv2.VideoWriter(output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
    if export_separate:
      outline_output_path = self.get_unique_filename(output_dir, 'outline', '.avi')
      outline_out = cv2.VideoWriter(outline_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      filled_output_path = self.get_unique_filename(output_dir, 'filled', '.avi')
      filled_out = cv2.VideoWriter(filled_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      center_val_output_path = self.get_unique_filename(output_dir, 'center_val', '.avi')
      center_val_out = cv2.VideoWriter(center_val_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      center_org_output_path = self.get_unique_filename(output_dir, 'center_org', '.avi')
      center_org_out = cv2.VideoWriter(center_org_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      connections_output_path = self.get_unique_filename(output_dir, 'connections', '.avi')
      connections_out = cv2.VideoWriter(connections_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      great_circle_output_path = self.get_unique_filename(output_dir, 'great_circle', '.avi')
      great_circle_out = cv2.VideoWriter(great_circle_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      center_ellipse_output_path = self.get_unique_filename(output_dir, 'center_ellipse', '.avi')
      center_ellipse_out = cv2.VideoWriter(center_ellipse_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      nr_val_output_path = self.get_unique_filename(output_dir, 'nr_val', '.avi')
      nr_val_out = cv2.VideoWriter(nr_val_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      nr_org_output_path = self.get_unique_filename(output_dir, 'nr_org', '.avi')
      nr_org_out = cv2.VideoWriter(nr_org_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      #connection_length_output_path = self.get_unique_filename(output_dir, 'connection_length', '.avi')
      #connection_length_out = cv2.VideoWriter(connection_length_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      circle_val_output_path = self.get_unique_filename(output_dir, 'circle_val', '.avi')
      circle_val_out = cv2.VideoWriter(circle_val_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      circle_org_output_path = self.get_unique_filename(output_dir, 'circle_org', '.avi')
      circle_org_out = cv2.VideoWriter(circle_org_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      #path_output_path = self.get_unique_filename(output_dir, 'path', '.avi')
      #path_out = cv2.VideoWriter(path_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))
      #all_paths_output_path = self.get_unique_filename(output_dir, 'all_paths', '.avi')
      #all_paths_out = cv2.VideoWriter(all_paths_output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))

    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.progress_bar['maximum'] = total_frames
    self.progress_bar['value'] = 0

    for frame_num in range(total_frames):
      if self.stop_processing:
        break

      ret, frame = self.cap.read()
      if not ret:
        break

      radius = int(self.radius_var.get() / 100 * (frame.shape[1] // 2))
      frame_with_boundaries, boundary_img, self.kmeans, filled_only_img = self.process_frame(
        frame, self.num_colors, self.selected_cluster, self.BOUNDARY_COLOR, radius,
        self.kmeans, self.grayscale_var.get(), self.fill_cluster_var.get()
      )

      if filled_only_img is None:
        _, _, _, filled_only_img = self.process_frame(
          frame, self.num_colors, self.selected_cluster, self.BOUNDARY_COLOR, radius,
          self.kmeans, self.grayscale_var.get(), True
        )

      second_centers = self.perform_second_kmeans(filled_only_img, self.second_kmeans_clusters_var.get())
      if len(second_centers) > 0:
        second_centers = self.merge_close_centers(second_centers, self.merge_threshold_var.get(), frame.shape[:2])
        self.update_centers(second_centers)
        if self.show_second_kmeans_var.get():
          self.draw_lines_and_markers(frame_with_boundaries, second_centers, self.CONNEC_VAL_COLOR, self.CONNEC_VAL_THICKNESS)
        self.draw_center_val(frame_with_boundaries, self.CENTER_VAL_COLOR, self.CENTER_VAL_THICKNESS)
        self.draw_center_org(frame_with_boundaries, self.CENTER_ORG_COLOR, self.CENTER_ORG_THICKNESS)
        self.draw_center_paths(frame_with_boundaries, self.PATH_VAL_COLOR, self.PATH_VAL_THICKNESS)

      if export_original:
        out.write(frame_with_boundaries)
      if export_separate:
        outline_out.write(boundary_img)
        filled_out.write(filled_only_img)

        center_val_img = np.zeros_like(frame)
        self.draw_center_val(center_val_img, self.EXPORT_CENTER_VAL_COLOR, self.EXPORT_CENTER_VAL_THICKNESS)
        center_val_out.write(center_val_img)

        center_org_img = np.zeros_like(frame)
        self.draw_center_markers(center_org_img, self.center_org, self.EXPORT_CENTER_ORG_COLOR, self.EXPORT_CENTER_ORG_THICKNESS)
        center_org_out.write(center_org_img)

        connections_img = np.zeros_like(frame)
        self.draw_connections_only(connections_img, self.EXPORT_CONNEC_VAL_COLOR, self.EXPORT_CONNEC_VAL_THICKNESS)
        connections_out.write(connections_img)
        
        great_circle_img = np.zeros_like(frame)
        self.draw_great_circle_only(great_circle_img, self.EXPORT_CONNEC_VAL_COLOR, self.EXPORT_CONNEC_VAL_THICKNESS)
        great_circle_out.write(great_circle_img)
        
        center_ellipse_img = np.zeros_like(frame)
        self.draw_center_ellipse_only(center_ellipse_img, self.EXPORT_CONNEC_VAL_COLOR, self.EXPORT_CONNEC_VAL_THICKNESS)
        center_ellipse_out.write(center_ellipse_img)

        nr_val_img = np.zeros_like(frame)
        self.draw_center_numbers(nr_val_img, self.EXPORT_NUMBER_VAL_COLOR, self.FONT_SIZE, self.FONT_THICKNESS, self.center_val)
        nr_val_out.write(nr_val_img)

        nr_org_img = np.zeros_like(frame)
        self.draw_center_numbers(nr_org_img, self.EXPORT_NUMBER_ORG_COLOR, self.FONT_SIZE, self.FONT_THICKNESS, self.center_org)
        nr_org_out.write(nr_org_img)

        #connection_length_img = np.zeros_like(frame)
        #self.draw_connection_lengths(connection_length_img, self.EXPORT_CONNEC_LENGTH_COLOR, self.FONT_SIZE, self.FONT_THICKNESS)
        #connection_length_out.write(connection_length_img)

        circle_val_img = np.zeros_like(frame)
        self.draw_center_circles(circle_val_img, self.EXPORT_CIRCLE_VAL_COLOR, self.EXPORT_CIRCLE_VAL_THICKNESS)
        circle_val_out.write(circle_val_img)

        circle_org_img = np.zeros_like(frame)
        self.draw_center_org_circles(circle_org_img, self.EXPORT_CIRCLE_ORG_COLOR, self.EXPORT_CIRCLE_ORG_THICKNESS)
        circle_org_out.write(circle_org_img)

        #path_img = np.zeros_like(frame)
        #self.draw_center_paths(path_img, self.EXPORT_PATH_VAL_COLOR, self.EXPORT_PATH_VAL_THICKNESS)
        #path_out.write(path_img)

        #all_paths_img = np.zeros_like(frame)
        #self.draw_all_paths(all_paths_img, self.EXPORT_PATH_VAL_COLOR, self.EXPORT_PATH_VAL_THICKNESS)
        #all_paths_out.write(all_paths_img)

      self.progress_bar['value'] = frame_num + 1
      self.progress_label.config(text=f"Exporting frame {frame_num + 1}/{total_frames}")
      self.root.update_idletasks()

    self.cap.release()
    if export_original:
      out.release()
    if export_separate:
      outline_out.release()
      filled_out.release()
      center_val_out.release()
      center_org_out.release()
      connections_out.release()
      great_circle_out.release()
      center_ellipse_out.release()
      nr_val_out.release()
      nr_org_out.release()
      #connection_length_out.release()
      circle_val_out.release()
      circle_org_out.release()
      #path_out.release()
      #all_paths_out.release()
    cv2.destroyAllWindows()

    if self.stop_processing:
      messagebox.showinfo("Info", "Video processing stopped.")
    else:
      messagebox.showinfo("Info", "Video processing completed and saved.")

  def draw_connection_lengths(self, image, color, font_size, thickness):
    radius = int(self.radius_var.get() / 100 * (self.preview_frame.shape[1] // 2))
    for i, center in enumerate(self.center_val):
      distances = np.linalg.norm(np.array(self.center_val) - center, axis=1)
      closest_indices = distances.argsort()[1:self.connection_count_var.get()+1]
      for idx in closest_indices:
        closest_center = self.center_val[idx]
        mid_point = ((int(center[1]) + int(closest_center[1])) // 2, (int(center[0]) + int(closest_center[0])) // 2)
        length = np.linalg.norm(np.array(center) - np.array(closest_center))
        hav_length = self.haversine_distance(center[:2], closest_center[:2])
        cv2.putText(image, f"{hav_length/radius:.3f} ({length/radius:.3f})", (mid_point[0], mid_point[1]),  
            cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.EXPORT_CONNEC_LENGTH_COLOR, self.FONT_THICKNESS)

  def select_video(self):
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if video_path:
      self.cap = cv2.VideoCapture(video_path)
      self.fps = self.cap.get(cv2.CAP_PROP_FPS)
      ret, self.preview_frame = self.cap.read()
      if not ret:
        messagebox.showerror("Error", "Failed to read video")
        return
      self.kmeans = None
      self.current_frame_index = 0
      self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.frame_slider.config(to=self.total_frames - 1)
      self.video_path_label.config(text=os.path.basename(video_path))
      self.update_preview()
    else:
      self.video_path_label.config(text="Bitte Video laden")

  def reset_parameters(self):
    self.num_colors = self.NUM_COLORS
    self.selected_cluster = self.SELECTED_CLUSTER
    self.grayscale_var.set(self.GRAYSCALE_VAR)
    self.radius_var.set(self.RADIUS_VAR)
    self.fill_cluster_var.set(self.FILL_CLUSTER_VAR)
    self.second_kmeans_clusters_var.set(self.SECOND_KMEANS_CLUSTERS_VAR)
    self.merge_threshold_var.set(self.MERGE_THRESHOLD_VAR)
    self.t_val_var.set(self.T_VAL_VAR)
    self.t_exist_var.set(self.T_EXIST_VAR)
    self.connection_count_var.set(self.CONNECTION_COUNT_VAR)
    self.show_cluster_number_var.set(self.SHOW_CLUSTER_NUMBER_VAR)
    self.show_second_kmeans_var.set(self.SHOW_SECOND_KMEANS_VAR)
    self.show_all_paths_var.set(self.SHOW_ALL_PATHS_VAR)
    self.show_connection_length_var.set(self.SHOW_CONNECTION_LENGTH_VAR)
    self.update_preview()

  def update_frame(self, index):
    if self.cap:
      self.current_frame_index = int(index)
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
      ret, self.preview_frame = self.cap.read()
      if not ret:
        return
      self.update_preview()

class GreatCircleSegments:
  def __init__(self, lon1, lat1, lon2, lat2, num_points=20):
    self.lon1 = lon1
    self.lat1 = lat1
    self.lon2 = lon2
    self.lat2 = lat2
    self.num_points = num_points
    self.points = self.compute_segments()

  def compute_segments(self):
    points = []
    start = Point(self.lat1, self.lon1)
    end = Point(self.lat2, self.lon2)
    total_distance = geodesic(start, end).kilometers
    initial_bearing = self.calculate_initial_compass_bearing(start, end)
    for i in range(self.num_points + 1):
      frac = i / self.num_points
      distance = total_distance * frac
      intermediate_point = geodesic(kilometers=distance).destination(point=start, bearing=initial_bearing)
      points.append((intermediate_point.longitude, intermediate_point.latitude))
    return points

  @staticmethod
  def calculate_initial_compass_bearing(pointA, pointB):
    import math

    lat1 = math.radians(pointA.latitude)
    lat2 = math.radians(pointB.latitude)
    diffLong = math.radians(pointB.longitude - pointA.longitude)

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

  def draw_great_circle(self, image, center_x, center_y, image_size, color):
    m = Basemap(projection='ortho', lat_0=0, lon_0=0)
    
    for i in range(len(self.points) - 1):
      lon_start, lat_start = self.points[i]
      lon_end, lat_end = self.points[i + 1]

      x_start, y_start = m(lon_start, lat_start)
      x_end, y_end = m(lon_end, lat_end)

      x_start, y_start = int(x_start / m.xmax * image_size), int((m.ymax - y_start) / m.ymax * image_size)
      x_end, y_end = int(x_end / m.xmax * image_size), int((m.ymax - y_end) / m.ymax * image_size)

      x_start += center_x - image_size // 2
      y_start += center_y - image_size // 2
      x_end += center_x - image_size // 2
      y_end += center_y - image_size // 2

      if 0 <= x_start < image.shape[1] and 0 <= y_start < image.shape[0] and 0 <= x_end < image.shape[1] and 0 <= y_end < image.shape[0]:
        cv2.line(image, (x_start, y_start), (x_end, y_end), color, 1)
    
    return image

class CenterEllipse:
  def __init__(self, center_x, center_y, x, angle, x_max, r, color, thikness):
    self.center_x = center_x
    self.center_y = center_y
    self.x = x
    self.angle = angle
    self.x_max = x_max
    self.r = r
    self.color = color
    self.thikness = thikness
    
  def calculate_params(self):
    x_normalized = self.x / self.x_max
    r1 = self.r - np.tan(x_normalized) * self.r
    r2 = self.r 
    return (max(1, 2*r1), max(1, 2*r2))

  def draw_ellipse(self, image):
    width, height = self.calculate_params()
    angle = self.angle
    cv2.ellipse(image, (self.center_x, self.center_y), (int(width), int(height)), angle, 0, 360, self.color, self.thikness)

if __name__ == "__main__":
  root = tk.Tk()
  app = VideoClusterFilterApp(root)
  root.mainloop()
