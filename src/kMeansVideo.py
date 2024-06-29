# Created 2024-06-29 by David Herren

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans, KMeans
from skimage.segmentation import mark_boundaries
import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk

stop_processing = False
center_org = []
center_val = []
t_val_frames = []
t_org_frames = []
center_paths = []

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
    fill_img[np.where(new_fill_area == (255))] = [int(c * 255) for c in boundary_color]
    
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
  global preview_frame, num_colors, selected_cluster, boundary_color, kmeans, grayscale_var, radius_var, fill_cluster_var, show_second_kmeans_var, second_kmeans_clusters_var, merge_threshold_var, t_val_var, show_cluster_number_var, connection_count_var, t_exist_var, previous_centers, center_org, center_val, t_val_frames, center_val_var, center_paths
  if preview_frame is not None:
    radius = int(radius_var.get() / 100 * (preview_frame.shape[1] // 2))
    preview_with_boundaries, _, kmeans, filled_only_img = process_frame(
      preview_frame, num_colors, selected_cluster, boundary_color, radius,
      kmeans, grayscale_var.get(), fill_cluster_var.get()
    )

    # Always generate the filled_only_img for second K-means
    if filled_only_img is None:
      _, _, _, filled_only_img = process_frame(
        preview_frame, num_colors, selected_cluster, boundary_color, radius,
        kmeans, grayscale_var.get(), True
      )
    second_kmeans_clusters = second_kmeans_clusters_var.get()
    second_centers = perform_second_kmeans(filled_only_img, second_kmeans_clusters)
    second_centers = merge_close_centers(second_centers, merge_threshold_var.get(), preview_frame.shape[:2])
    update_centers(second_centers, merge_threshold_var.get(), t_val_var.get(), t_exist_var.get(), second_kmeans_clusters)

    if show_second_kmeans_var.get():
      draw_lines_and_markers(preview_with_boundaries, second_centers, connection_count_var.get(), merge_threshold_var.get())

    if center_val_var.get():
      draw_center_val(preview_with_boundaries, center_val, merge_threshold_var.get())
      draw_center_val_paths(preview_with_boundaries, center_val, center_paths, merge_threshold_var.get())

    preview_image = Image.fromarray(cv2.cvtColor(preview_with_boundaries, cv2.COLOR_BGR2RGB))
    preview_photo = ImageTk.PhotoImage(preview_image)
    preview_label.config(image=preview_photo)
    preview_label.image = preview_photo

    num_colors_label.config(text=str(num_colors))
    selected_cluster_label.config(text=str(selected_cluster))

def draw_lines_and_markers(image, centers, connection_count, merge_threshold):
  for i, center in enumerate(centers):
    cv2.drawMarker(image, (int(center[1]), int(center[0])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.circle(image, (int(center[1]), int(center[0])), int(merge_threshold / 100 * (min(image.shape[:2]) / 2)), (0, 255, 0), 2)
  for i, center in enumerate(centers):
    distances = np.linalg.norm(centers - center, axis=1)
    closest_indices = distances.argsort()[1:connection_count+1]
    for idx in closest_indices:
      closest_center = centers[idx]
      cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), (0, 255, 0), 2)

def draw_center_val(image, centers, merge_threshold):
  for i, center in enumerate(centers):
    cv2.drawMarker(image, (int(center[1]), int(center[0])), color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.circle(image, (int(center[1]), int(center[0])), int(merge_threshold / 100 * (min(image.shape[:2]) / 2)), (255, 0, 255), 2)
    if show_cluster_number_var.get():
      cv2.putText(image, str(i), (int(center[1]) + 10, int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    distances = np.linalg.norm(np.array(centers) - center, axis=1)
    closest_indices = distances.argsort()[1:connection_count_var.get()+1]
    for idx in closest_indices:
      closest_center = centers[idx]
      cv2.line(image, (int(center[1]), int(center[0])), (int(closest_center[1]), int(closest_center[0])), (255, 0, 255), 2)

def draw_center_val_paths(image, centers, paths, merge_threshold):
  path_color = (255, 255, 0)
  for i, center in enumerate(centers):
    # Draw the center marker and circle
    cv2.drawMarker(image, (int(center[1]), int(center[0])), color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.circle(image, (int(center[1]), int(center[0])), int(merge_threshold / 100 * (min(image.shape[:2]) / 2)), (255, 0, 255), 2)
    
    # Draw the cluster number
    if show_cluster_number_var.get():
      cv2.putText(image, str(i), (int(center[1]) + 10, int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    # Draw the path
    if len(paths[i]) > 1:
      for j in range(1, len(paths[i])):
        cv2.line(image, (int(paths[i][j-1][1]), int(paths[i][j-1][0])), (int(paths[i][j][1]), int(paths[i][j][0])), path_color, 2)

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

def update_centers(new_centers, merge_threshold, t_val, t_exist, max_centers):
  global center_org, center_val, t_val_frames, t_org_frames, center_paths
  if len(center_val) == 0:
    center_val = new_centers.tolist()[:max_centers]  # Limit to max_centers
    t_val_frames = [t_val] * len(center_val)
    center_paths = [[center] for center in center_val]  # Initialize paths
    return

  # Update existing center_val with new_centers
  for new_center in new_centers:
    merged = False
    for i, val_center in enumerate(center_val):
      if np.linalg.norm(new_center[:2] - val_center[:2]) < merge_threshold:
        center_val[i] = (np.array(val_center) * (t_val_frames[i] - 1) + np.array(new_center)) / t_val_frames[i]
        t_val_frames[i] = t_val
        center_paths[i].append(new_center)  # Add to path
        merged = True
        break
    if not merged:
      center_org.append(new_center.tolist())
      t_org_frames.append(t_exist)

  # Merge overlapping center_val
  i = 0
  while i < len(center_val):
    j = i + 1
    while j < len(center_val):
      if np.linalg.norm(np.array(center_val[i][:2]) - np.array(center_val[j][:2])) < merge_threshold:
        center_val[i] = (np.array(center_val[i]) + np.array(center_val[j])) / 2
        center_paths[i].extend(center_paths[j])  # Merge paths
        del center_val[j]
        del t_val_frames[j]
        del center_paths[j]
      else:
        j += 1
    i += 1

  # Apply moving average and remove old centers
  for i in range(len(center_val)):
    if t_val_frames[i] > 1:
      center_val[i] = (np.array(center_val[i]) * (t_val_frames[i] - 1) + np.array(center_val[i])) / t_val_frames[i]
    t_val_frames[i] -= 1

  # Remove centers that have not been updated for t_val frames
  indices_to_keep = [i for i in range(len(center_val)) if t_val_frames[i] > 0]
  center_val = [center_val[i] for i in indices_to_keep]
  t_val_frames = [t_val_frames[i] for i in indices_to_keep]
  center_paths = [center_paths[i] for i in indices_to_keep]

  # Process new centers that have existed for t_exist frames
  new_stable_centers = []
  new_t_val_frames = []
  new_paths = []
  for i in range(len(center_org)):
    t_org_frames[i] -= 1
    if t_org_frames[i] <= 0:
      existing = False
      for j, val_center in enumerate(center_val):
        if np.linalg.norm(np.array(center_org[i][:2]) - np.array(val_center[:2])) < merge_threshold:
          existing = True
          break
      if not existing:
        new_stable_centers.append(center_org[i])
        new_t_val_frames.append(t_val)
        new_paths.append([center_org[i]])

  center_val.extend(new_stable_centers)
  t_val_frames.extend(new_t_val_frames)
  center_paths.extend(new_paths)
  indices_to_keep = [i for i in range(len(center_org)) if t_org_frames[i] > 0]
  center_org = [center_org[i] for i in indices_to_keep]
  t_org_frames = [t_org_frames[i] for i in indices_to_keep]

  # Merge overlapping center_val again to ensure no overlap
  i = 0
  while i < len(center_val):
    j = i + 1
    while j < len(center_val):
      if np.linalg.norm(np.array(center_val[i][:2]) - np.array(center_val[j][:2])) < merge_threshold:
        center_val[i] = (np.array(center_val[i]) + np.array(center_val[j])) / 2
        center_paths[i].extend(center_paths[j])  # Merge paths
        del center_val[j]
        del t_val_frames[j]
        del center_paths[j]
      else:
        j += 1
    i += 1

  # Limit the number of center_val to max_centers
  if len(center_val) > max_centers:
    center_val = center_val[:max_centers]
    t_val_frames = t_val_frames[:max_centers]
    center_paths = center_paths[:max_centers]

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
  global cap, num_colors, selected_cluster, boundary_color, kmeans, grayscale_var, radius_var, fill_cluster_var, second_kmeans_clusters_var, merge_threshold_var, t_val_var, t_exist_var, previous_centers, connection_count_var, center_org, center_val, t_val_frames, t_org_frames

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

    if filled_only_img is None:
      _, _, _, filled_only_img = process_frame(
        frame, num_colors, selected_cluster, boundary_color, radius,
        kmeans, grayscale_var.get(), True
      )

    second_kmeans_clusters = second_kmeans_clusters_var.get()
    second_centers = perform_second_kmeans(filled_only_img, second_kmeans_clusters)
    if len(second_centers) > 0:
      second_centers = merge_close_centers(second_centers, merge_threshold_var.get(), frame.shape[:2])
      update_centers(second_centers, merge_threshold_var.get(), t_val_var.get(), t_exist_var.get(), second_kmeans_clusters)
      if show_second_kmeans_var.get():
        draw_lines_and_markers(frame_with_boundaries, second_centers, connection_count_var.get(), merge_threshold_var.get())
      draw_center_val(frame_with_boundaries, center_val, merge_threshold_var.get())
      draw_center_val_paths(frame_with_boundaries, center_val, center_paths, merge_threshold_var.get())

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
  global cap, preview_frame, kmeans, current_frame_index, total_frames, previous_centers, center_ids
  video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
  if video_path:
    cap = cv2.VideoCapture(video_path)
    ret, preview_frame = cap.read()
    if not ret:
      messagebox.showerror("Error", "Failed to read video")
      return
    kmeans = None
    previous_centers = None
    center_ids = []
    current_frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_slider.config(to=total_frames - 1)
    video_path_label.config(text=os.path.basename(video_path))  # Zeigt nur den Namen des Videos an
    update_preview()
  else:
    video_path_label.config(text="Bitte Video laden")  # Zeigt "Bitte Video laden" an, wenn kein Video geladen ist

def reset_parameters():
  global num_colors, selected_cluster, boundary_color, grayscale_var, radius_var, fill_cluster_var, second_kmeans_clusters_var, merge_threshold_var, t_val_var, connection_count_var, t_exist_var, show_cluster_number_var, show_second_kmeans_var
  num_colors = 4
  selected_cluster = 0
  boundary_color = (0, 1, 0)
  grayscale_var.set(1)
  radius_var.set(80)
  fill_cluster_var.set(0)
  second_kmeans_clusters_var.set(4)
  merge_threshold_var.set(20)
  t_val_var.set(5)
  t_exist_var.set(5)
  connection_count_var.set(1)
  show_cluster_number_var.set(1)
  show_second_kmeans_var.set(1)
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
boundary_color = (0, 1, 0)
preview_frame = None
current_frame_index = 0
total_frames = 0
previous_centers = None
center_ids = []

# Initialize the K-means algorithm globally
kmeans = None

# Create a Tkinter window for parameter adjustment
root = tk.Tk()
root.title("k-Means Video Cluster Filter")

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
btn_select_video.grid(row=0, column=0, pady=5)

btn_reset = tk.Button(right_frame, text="Reset", command=reset_parameters)
btn_reset.grid(row=0, column=1, pady=5)

video_path_label = tk.Label(right_frame, text="No Video is Loaded!")
video_path_label.grid(row=0, column=3, columnspan=2, pady=2)

# Create buttons to adjust parameters and display current values
tk.Label(right_frame, text="Clusters").grid(row=2, column=0, pady=2)
num_colors_label = tk.Label(right_frame, text=str(num_colors))
num_colors_label.grid(row=2, column=1, pady=2)

btn_increase_colors = tk.Button(right_frame, text="+", command=increase_colors)
btn_increase_colors.grid(row=3, column=0, pady=2)
btn_decrease_colors = tk.Button(right_frame, text="-", command=decrease_colors)
btn_decrease_colors.grid(row=3, column=1, pady=2)

tk.Label(right_frame, text="Boundary").grid(row=4, column=0, pady=2)
selected_cluster_label = tk.Label(right_frame, text=str(selected_cluster))
selected_cluster_label.grid(row=4, column=1, pady=2)

btn_increase_cluster = tk.Button(right_frame, text="+", command=increase_cluster)
btn_increase_cluster.grid(row=5, column=0, pady=2)
btn_decrease_cluster = tk.Button(right_frame, text="-", command=decrease_cluster)
btn_decrease_cluster.grid(row=5, column=1, pady=2)

# Create slider to adjust radius
tk.Label(right_frame, text="Radius (%)").grid(row=7, column=0, pady=2, sticky='w')
radius_var = tk.IntVar(value=80)
tk.Scale(right_frame, from_=0, to=100, orient='horizontal', variable=radius_var, command=update_radius).grid(row=7, column=1, pady=2, sticky='w')

# Create slider to adjust the number of clusters for the second K-means
tk.Label(right_frame, text="Second K-means Clusters").grid(row=8, column=0, pady=2, sticky='w')
second_kmeans_clusters_var = tk.IntVar(value=4)
tk.Scale(right_frame, from_=1, to=20, orient='horizontal', variable=second_kmeans_clusters_var, command=lambda x: update_preview()).grid(row=8, column=1, pady=2, sticky='w')

# Create slider to adjust the merge threshold for second K-means
tk.Label(right_frame, text="Merge Threshold (%)").grid(row=9, column=0, pady=2, sticky='w')
merge_threshold_var = tk.IntVar(value=20)
tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=merge_threshold_var, command=lambda x: update_preview()).grid(row=9, column=1, pady=2, sticky='w')

# Create slider to adjust t_val
tk.Label(right_frame, text="t_val (frames)").grid(row=10, column=0, pady=2, sticky='w')
t_val_var = tk.IntVar(value=10)
tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=t_val_var, command=lambda x: update_preview()).grid(row=10, column=1, pady=2, sticky='w')

# Create slider to adjust t_exist
tk.Label(right_frame, text="t_exist (frames)").grid(row=11, column=0, pady=2, sticky='w')
t_exist_var = tk.IntVar(value=20)
tk.Scale(right_frame, from_=1, to=100, orient='horizontal', variable=t_exist_var, command=lambda x: update_preview()).grid(row=11, column=1, pady=2, sticky='w')

# Create slider to adjust the number of connections between centers
tk.Label(right_frame, text="Number of Connections").grid(row=12, column=0, pady=2, sticky='w')
connection_count_var = tk.IntVar(value=0)
tk.Scale(right_frame, from_=0, to=4, orient='horizontal', variable=connection_count_var, command=lambda x: update_preview()).grid(row=12, column=1, pady=2, sticky='w')

# Create checkboxes to select options
export_original_var = tk.IntVar(value=1)
export_separate_var = tk.IntVar(value=0)
grayscale_var = tk.IntVar(value=1)
fill_cluster_var = tk.IntVar(value=0)
show_second_kmeans_var = tk.IntVar(value=1)
center_val_var = tk.IntVar(value=1)
show_cluster_number_var = tk.IntVar(value=1)

chk_export_original = tk.Checkbutton(right_frame, text="Export Original Video", variable=export_original_var)
chk_export_original.grid(row=13, column=0, sticky='w', pady=2)

chk_export_separate = tk.Checkbutton(right_frame, text="Export All Layers Separately", variable=export_separate_var)
chk_export_separate.grid(row=14, column=0, sticky='w', pady=2)

chk_grayscale = tk.Checkbutton(right_frame, text="Display Grayscale Video", variable=grayscale_var, command=toggle_grayscale)
chk_grayscale.grid(row=15, column=0, sticky='w', pady=2)

chk_fill_cluster = tk.Checkbutton(right_frame, text="Fill Selected Cluster", variable=fill_cluster_var, command=update_preview)
chk_fill_cluster.grid(row=16, column=0, sticky='w', pady=2)

chk_show_second_kmeans = tk.Checkbutton(right_frame, text="Show Second K-means", variable=show_second_kmeans_var, command=update_preview)
chk_show_second_kmeans.grid(row=17, column=0, sticky='w', pady=2)

chk_center_val = tk.Checkbutton(right_frame, text="Show Center Val", variable=center_val_var, command=update_preview)
chk_center_val.grid(row=18, column=0, sticky='w', pady=2)

chk_show_cluster_number = tk.Checkbutton(right_frame, text="Show Cluster Number", variable=show_cluster_number_var, command=update_preview)  # Checkbox for showing cluster number
chk_show_cluster_number.grid(row=19, column=0, sticky='w', pady=2)

# Add frame slider
frame_slider = tk.Scale(right_frame, from_=0, to=total_frames - 1, orient='horizontal', command=update_frame)
frame_slider.grid(row=20, column=0, columnspan=4, pady=2, sticky="ew")

# Add a progress bar and a label to show the current frame
progress_bar = ttk.Progressbar(right_frame, orient='horizontal', mode='determinate')
progress_bar.grid(row=21, column=0, columnspan=2, pady=20, sticky="ew")
progress_label = tk.Label(right_frame, text="Frame 0/0")
progress_label.grid(row=21, column=2, columnspan=2, pady=2, sticky="ew")

# Buttons to start, stop, and exit
btn_start_processing = tk.Button(right_frame, text="Export Video", command=start_processing)
btn_start_processing.grid(row=22, column=0, pady=2, sticky='ew')

btn_stop_processing = tk.Button(right_frame, text="Stop", command=stop_processing)
btn_stop_processing.grid(row=22, column=1, pady=2, sticky='ew')

btn_exit = tk.Button(right_frame, text="Exit", command=root.destroy)
btn_exit.grid(row=23, column=3, pady=2, sticky='ew')

root.mainloop()
