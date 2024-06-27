# Created 2024-06-27 by David Herren

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans, KMeans
from skimage.segmentation import mark_boundaries
import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk

stop_processing = False  # Global variable to handle stop request

def get_unique_filename(base_path, base_name, ext):
  id = 0
  while os.path.exists(f"{base_path}/{base_name}_{id}{ext}"):
    id += 1
  return f"{base_path}/{base_name}_{id}{ext}"

def process_frame(frame, num_colors, selected_cluster, boundary_color, radius, kmeans=None, grayscale=False, fill_cluster=False):
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
    # Create a new filled area by subtracting the cluster mask from the circle mask
    new_fill_area = mask.copy()
    new_fill_area[mask_labels.reshape(image_np.shape[:2])] = 0
    fill_img = np.zeros_like(frame)
    fill_img[np.where(new_fill_area == 255)] = [int(c * 255) for c in boundary_color]
    
    # Create inverted filled area
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

def update_preview():
  global preview_frame, num_colors, selected_cluster, boundary_color, kmeans, grayscale_var, radius_var, fill_cluster_var, second_kmeans_var, second_kmeans_clusters_var, merge_threshold_var, smoothing_factor_var
  if preview_frame is not None:
    radius = int(radius_var.get() / 100 * (preview_frame.shape[1] // 2))
    preview_with_boundaries, _, kmeans, filled_only_img = process_frame(
      preview_frame, num_colors, selected_cluster, boundary_color, radius,
      kmeans, grayscale_var.get(), fill_cluster_var.get()
    )
    
    if second_kmeans_var.get():
      # Always generate the filled_only_img for second K-means, but don't show it in the preview
      if filled_only_img is None:
        _, _, _, filled_only_img = process_frame(
          preview_frame, num_colors, selected_cluster, boundary_color, radius,
          kmeans, grayscale_var.get(), True
        )
      second_kmeans_clusters = second_kmeans_clusters_var.get()
      second_centers = perform_second_kmeans(filled_only_img, second_kmeans_clusters)
      second_centers = merge_close_centers(second_centers, merge_threshold_var.get(), preview_frame.shape[:2])
      second_centers = apply_smoothing_filter(second_centers, smoothing_factor_var.get())
      draw_lines_and_markers(preview_with_boundaries, second_centers)
    
    preview_image = Image.fromarray(cv2.cvtColor(preview_with_boundaries, cv2.COLOR_BGR2RGB))
    preview_photo = ImageTk.PhotoImage(preview_image)
    preview_label.config(image=preview_photo)
    preview_label.image = preview_photo

    num_colors_label.config(text=str(num_colors))
    selected_cluster_label.config(text=str(selected_cluster))

def draw_lines_and_markers(image, centers):
  for center in centers:
    cv2.drawMarker(image, (int(center[1]), int(center[0])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
  for i, center in enumerate(centers):
    distances = np.linalg.norm(centers - center, axis=1)
    closest_indices = distances.argsort()[1:3]
    for idx in closest_indices:
      closest_center = centers[idx]
      cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), (0, 255, 0), 1)

def decrease_colors():
  global num_colors, selected_cluster, kmeans
  num_colors = max(1, num_colors - 1)
  if selected_cluster >= num_colors:
    selected_cluster = num_colors - 1
  kmeans = None  # Reset kmeans to ensure re-initialization with the new number of clusters
  update_preview()

def increase_colors():
  global num_colors, kmeans
  num_colors += 1
  kmeans = None  # Reset kmeans to ensure re-initialization with the new number of clusters
  update_preview()

def decrease_cluster():
  global selected_cluster
  selected_cluster = max(0, selected_cluster - 1)
  update_preview()

def increase_cluster():
  global selected_cluster
  if selected_cluster < num_colors - 1:
    selected_cluster += 1
    update_preview()

def update_boundary_color():
  global boundary_color
  r = boundary_r_var.get() / 255.0
  g = boundary_g_var.get() / 255.0
  b = boundary_b_var.get() / 255.0
  boundary_color = (r, g, b)
  update_preview()

def toggle_grayscale():
  update_preview()

def update_radius(val):
  update_preview()

def perform_second_kmeans(image, n_clusters):
  height, width, _ = image.shape
  Y, X = np.ogrid[:height, :width]
  mask = np.any(image != [0, 0, 0], axis=-1)  # Only use filled pixels
  positions = np.column_stack(np.where(mask))

  if positions.size == 0:
    return []

  # Image data only for filled pixels
  data = image[mask].reshape(-1, 3)

  if data.size == 0:
    return []

  # Combine positions and image data
  data = np.hstack((positions, data))

  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
  centers = kmeans.cluster_centers_
  return centers

def merge_close_centers(centers, threshold_percent, image_shape):
  threshold = (threshold_percent / 100) * (min(image_shape) / 2)
  merged_centers = []
  while len(centers) > 0:
    center = centers[0]
    distances = np.linalg.norm(centers - center, axis=1)
    close_centers = centers[distances < threshold]
    centers = centers[distances >= threshold]
    merged_center = np.mean(close_centers, axis=0)
    merged_centers.append(merged_center)
  return np.array(merged_centers)

def apply_smoothing_filter(centers, smoothing_factor):
  smoothed_centers = np.copy(centers)
  for i in range(len(centers)):
    neighbors = []
    for j in range(len(centers)):
      if i != j:
        distance = np.linalg.norm(centers[i] - centers[j])
        if distance < smoothing_factor:  # Verwende Nachbarn innerhalb des glättungsfaktors
          neighbors.append(centers[j])
    if neighbors:
      neighbors = np.array(neighbors)
      smoothed_centers[i] = np.mean(neighbors, axis=0) * 0.5 + centers[i] * 0.5  # Kombiniere aktuelle und geglättete Position
  return smoothed_centers

def start_processing():
  global stop_processing
  stop_processing = False
  export_original = export_original_var.get()
  export_separate = export_separate_var.get()
  output_dir = filedialog.askdirectory(title="Select Output Directory")
  if output_dir:
    process_video(export_original, export_separate, output_dir)

def stop_processing():
  global stop_processing
  stop_processing = True

def process_video(export_original, export_separate, output_dir):
  global cap, num_colors, selected_cluster, boundary_color, kmeans, grayscale_var, radius_var, fill_cluster_var, second_kmeans_clusters_var, merge_threshold_var, smoothing_factor_var

  # Initialize video writers
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  if export_original:
    output_path = get_unique_filename(output_dir, 'output', '.avi')
    out = cv2.VideoWriter(output_path, fourcc, 18.0, (int(cap.get(3)), int(cap.get(4))))
  if export_separate:
    outline_output_path = get_unique_filename(output_dir, 'outline', '.avi')
    outline_out = cv2.VideoWriter(outline_output_path, fourcc, 18.0, (int(cap.get(3)), int(cap.get(4))))
    filled_output_path = get_unique_filename(output_dir, 'filled', '.avi')
    filled_out = cv2.VideoWriter(filled_output_path, fourcc, 18.0, (int(cap.get(3)), int(cap.get(4))))
    second_kmeans_output_path = get_unique_filename(output_dir, 'second_kmeans', '.avi')
    second_kmeans_out = cv2.VideoWriter(second_kmeans_output_path, fourcc, 18.0, (int(cap.get(3)), int(cap.get(4))))

  # Rewind the video to the start
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  # Calculate the total number of frames in the video
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Configure the progress bar
  progress_bar['maximum'] = total_frames
  progress_bar['value'] = 0

  # Use the K-means algorithm initialized during the preview
  for frame_num in range(total_frames):
    if stop_processing:
      break

    ret, frame = cap.read()
    if not ret:
      break

    # Process the frame
    radius = int(radius_var.get() / 100 * (frame.shape[1] // 2))
    frame_with_boundaries, boundary_img, kmeans, filled_only_img = process_frame(
      frame, num_colors, selected_cluster, boundary_color, radius,
      kmeans, grayscale_var.get(), fill_cluster_var.get()
    )

    if second_kmeans_var.get() and filled_only_img is None:
      _, _, _, filled_only_img = process_frame(
        frame, num_colors, selected_cluster, boundary_color, radius,
        kmeans, grayscale_var.get(), True
      )

    if second_kmeans_var.get():
      second_kmeans_clusters = second_kmeans_clusters_var.get()
      second_centers = perform_second_kmeans(filled_only_img, second_kmeans_clusters)
      if len(second_centers) > 0:
        second_centers = merge_close_centers(second_centers, merge_threshold_var.get(), frame.shape[:2])
        second_centers = apply_smoothing_filter(second_centers, smoothing_factor_var.get())
        draw_lines_and_markers(frame_with_boundaries, second_centers)

    # Write the frames to the videos
    if export_original:
      out.write(frame_with_boundaries)
    if export_separate:
      outline_out.write(boundary_img)
      filled_out.write(filled_only_img)
      second_kmeans_out.write(filled_only_img)

    # Update progress bar
    progress_bar['value'] = frame_num + 1
    progress_label.config(text=f"Exporting frame {frame_num + 1}/{total_frames}")
    root.update_idletasks()

  # Release video objects
  cap.release()
  if export_original:
    out.release()
  if export_separate:
    outline_out.release()
    filled_out.release()
    second_kmeans_out.release()
  cv2.destroyAllWindows()

  if stop_processing:
    messagebox.showinfo("Info", "Video processing stopped.")
  else:
    messagebox.showinfo("Info", "Video processing completed and saved.")

def select_video():
  global cap, preview_frame, kmeans, current_frame_index, total_frames
  video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
  if video_path:
    cap = cv2.VideoCapture(video_path)
    ret, preview_frame = cap.read()
    if not ret:
      messagebox.showerror("Error", "Failed to read video")
      return
    kmeans = None
    current_frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_slider.config(to=total_frames - 1)
    update_preview()

def next_frame():
  global cap, preview_frame, current_frame_index
  if cap and current_frame_index < total_frames - 1:
    current_frame_index += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, preview_frame = cap.read()
    if not ret:
      current_frame_index -= 1
      return
    frame_slider.set(current_frame_index)
    update_preview()

def prev_frame():
  global cap, preview_frame, current_frame_index
  if cap and current_frame_index > 0:
    current_frame_index -= 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, preview_frame = cap.read()
    if not ret:
      current_frame_index += 1
      return
    frame_slider.set(current_frame_index)
    update_preview()

def update_frame(index):
  global cap, preview_frame, current_frame_index
  if cap:
    current_frame_index = int(index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, preview_frame = cap.read()
    if not ret:
      return
    update_preview()

# Initial parameters for k-means
num_colors = 4
selected_cluster = 0
boundary_color = (1, 0, 0)
preview_frame = None
current_frame_index = 0
total_frames = 0

# Initialize the K-means algorithm globally
kmeans = None

# Create a Tkinter window for parameter adjustment
root = tk.Tk()
root.title("K-means Parameter Adjustment")

# Create a frame for the preview image and parameters
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label to display the preview image
preview_label = tk.Label(left_frame)
preview_label.pack()

# Create a frame for the buttons and current values
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.BOTH, expand=True)

# Add a button to select the video file
btn_select_video = tk.Button(right_frame, text="Select Video", command=select_video)
btn_select_video.grid(row=0, column=0, columnspan=2, pady=5)

# Create buttons to adjust parameters and display current values
tk.Label(right_frame, text="Clusters").grid(row=1, column=0, pady=2)
num_colors_label = tk.Label(right_frame, text=str(num_colors))
num_colors_label.grid(row=1, column=1, pady=2)

btn_increase_colors = tk.Button(right_frame, text="+", command=increase_colors)
btn_increase_colors.grid(row=2, column=0, pady=2)
btn_decrease_colors = tk.Button(right_frame, text="-", command=decrease_colors)
btn_decrease_colors.grid(row=2, column=1, pady=2)

tk.Label(right_frame, text="Boundary").grid(row=1, column=2, pady=2)
selected_cluster_label = tk.Label(right_frame, text=str(selected_cluster))
selected_cluster_label.grid(row=1, column=3, pady=2)

btn_increase_cluster = tk.Button(right_frame, text="+", command=increase_cluster)
btn_increase_cluster.grid(row=2, column=2, pady=2)
btn_decrease_cluster = tk.Button(right_frame, text="-", command=decrease_cluster)
btn_decrease_cluster.grid(row=2, column=3, pady=2)

# Create sliders to adjust boundary color
tk.Label(right_frame, text="Boundary Color (RGB)").grid(row=3, column=0, columnspan=4, pady=2)
tk.Label(right_frame, text="R").grid(row=4, column=0, sticky='w')
boundary_r_var = tk.IntVar(value=int(boundary_color[2] * 255))
tk.Scale(right_frame, from_=0, to=255, orient='horizontal', variable=boundary_r_var, command=lambda x: update_boundary_color()).grid(row=4, column=1, pady=2)

tk.Label(right_frame, text="G").grid(row=4, column=2, sticky='w')
boundary_g_var = tk.IntVar(value=int(boundary_color[1] * 255))
tk.Scale(right_frame, from_=0, to=255, orient='horizontal', variable=boundary_g_var, command=lambda x: update_boundary_color()).grid(row=4, column=3, pady=2)

tk.Label(right_frame, text="B").grid(row=5, column=0, sticky='w')
boundary_b_var = tk.IntVar(value=int(boundary_color[0] * 255))
tk.Scale(right_frame, from_=0, to=255, orient='horizontal', variable=boundary_b_var, command=lambda x: update_boundary_color()).grid(row=5, column=1, pady=2)

# Create slider to adjust radius
tk.Label(right_frame, text="Radius (%)").grid(row=6, column=0, columnspan=4, pady=2)
radius_var = tk.IntVar(value=80)
tk.Scale(right_frame, from_=0, to=100, orient='horizontal', variable=radius_var, command=update_radius).grid(row=7, column=0, columnspan=4, pady=2)

# Create checkboxes to select options
export_original_var = tk.IntVar(value=1)
export_separate_var = tk.IntVar(value=0)
grayscale_var = tk.IntVar(value=0)
fill_cluster_var = tk.IntVar(value=0)
second_kmeans_var = tk.IntVar(value=0)

chk_export_original = tk.Checkbutton(right_frame, text="Export Original Video", variable=export_original_var)
chk_export_original.grid(row=8, column=0, columnspan=4, pady=2)

chk_export_separate = tk.Checkbutton(right_frame, text="Export All Layers Separately", variable=export_separate_var)
chk_export_separate.grid(row=9, column=0, columnspan=4, pady=2)

chk_grayscale = tk.Checkbutton(right_frame, text="Display Grayscale Video", variable=grayscale_var, command=toggle_grayscale)
chk_grayscale.grid(row=10, column=0, columnspan=4, pady=2)

chk_fill_cluster = tk.Checkbutton(right_frame, text="Fill Selected Cluster", variable=fill_cluster_var, command=update_preview)
chk_fill_cluster.grid(row=11, column=0, columnspan=4, pady=2)

chk_second_kmeans = tk.Checkbutton(right_frame, text="Second K-means", variable=second_kmeans_var, command=update_preview)
chk_second_kmeans.grid(row=12, column=0, columnspan=4, pady=2)

# Create slider to adjust the number of clusters for the second K-means
tk.Label(right_frame, text="Second K-means Clusters").grid(row=13, column=0, columnspan=4, pady=2)
second_kmeans_clusters_var = tk.IntVar(value=4)
tk.Scale(right_frame, from_=1, to=20, orient='horizontal', variable=second_kmeans_clusters_var, command=lambda x: update_preview()).grid(row=14, column=0, columnspan=4, pady=2)

# Create slider to adjust the merge threshold for second K-means
tk.Label(right_frame, text="Merge Threshold (%)").grid(row=15, column=0, columnspan=4, pady=2)
merge_threshold_var = tk.IntVar(value=10)
tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=merge_threshold_var, command=lambda x: update_preview()).grid(row=16, column=0, columnspan=4, pady=2)

# Create slider to adjust the smoothing factor
tk.Label(right_frame, text="Smoothing Factor").grid(row=17, column=0, columnspan=4, pady=2)
smoothing_factor_var = tk.IntVar(value=1)
tk.Scale(right_frame, from_=0, to=100, orient='horizontal', variable=smoothing_factor_var, command=lambda x: update_preview()).grid(row=18, column=0, columnspan=4, pady=2)

btn_start_processing = tk.Button(right_frame, text="Export Video", command=start_processing)
btn_start_processing.grid(row=19, column=0, columnspan=2, pady=2)
btn_stop_processing = tk.Button(right_frame, text="Stop", command=stop_processing)
btn_stop_processing.grid(row=19, column=2, columnspan=2, pady=2)

# Add a progress bar and a label to show the current frame
progress_bar = ttk.Progressbar(right_frame, orient='horizontal', mode='determinate')
progress_bar.grid(row=20, column=0, columnspan=4, pady=2, sticky="ew")
progress_label = tk.Label(right_frame, text="Frame 0/0")
progress_label.grid(row=21, column=0, columnspan=4, pady=2)

# Add frame slider
frame_slider = tk.Scale(right_frame, from_=0, to=total_frames - 1, orient='horizontal', command=update_frame)
frame_slider.grid(row=22, column=0, columnspan=4, pady=2, sticky="ew")

btn_prev_frame = tk.Button(right_frame, text="Previous Frame", command=prev_frame)
btn_prev_frame.grid(row=23, column=0, columnspan=2, pady=2)

btn_next_frame = tk.Button(right_frame, text="Next Frame", command=next_frame)
btn_next_frame.grid(row=23, column=2, columnspan=2, pady=2)

btn_exit = tk.Button(right_frame, text="Exit", command=root.destroy)
btn_exit.grid(row=24, column=0, columnspan=4, pady=2)

root.mainloop()
