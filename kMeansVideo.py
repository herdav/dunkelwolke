# Created 2024-06-26 by David Herren

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk

def get_unique_filename(base_path, base_name, ext):
  id = 0
  while os.path.exists(f"{base_path}/{base_name}_{id}{ext}"):
    id += 1
  return f"{base_path}/{base_name}_{id}{ext}"

def process_frame(frame, num_colors, selected_cluster, kmeans=None):
  image = cv2.resize(frame, (900, 900))
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
  mask = (labels == selected_cluster)
  masked_labels = np.where(mask.reshape(image_np.shape[:2]), labels.reshape(image_np.shape[:2]), -1)

  boundary_img = mark_boundaries(np.zeros_like(segmented_img), masked_labels, color=(0, 1, 0))
  boundary_img = (boundary_img * 255).astype(np.uint8)
  boundary_img = cv2.resize(boundary_img, (frame.shape[1], frame.shape[0]))

  return cv2.addWeighted(frame, 1, boundary_img, 1, 0), boundary_img, kmeans

def update_preview():
  global preview_frame, num_colors, selected_cluster, kmeans
  if preview_frame is not None:
    preview_with_boundaries, _, kmeans = process_frame(preview_frame, num_colors, selected_cluster, kmeans)
    preview_image = Image.fromarray(cv2.cvtColor(preview_with_boundaries, cv2.COLOR_BGR2RGB))
    preview_photo = ImageTk.PhotoImage(preview_image)
    preview_label.config(image=preview_photo)
    preview_label.image = preview_photo
    num_colors_label.config(text=str(num_colors))
    selected_cluster_label.config(text=str(selected_cluster))

def decrease_colors():
  global num_colors, kmeans
  num_colors = max(1, num_colors - 1)
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
  selected_cluster += 1
  update_preview()

def start_processing():
  export_original = export_original_var.get()
  export_boundary = export_boundary_var.get()
  output_dir = filedialog.askdirectory(title="Select Output Directory")
  if output_dir:
    process_video(export_original, export_boundary, output_dir)

def process_video(export_original, export_boundary, output_dir):
  global cap, num_colors, selected_cluster, kmeans

  # Initialize video writers
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  if export_original:
    output_path = get_unique_filename(output_dir, 'output', '.avi')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
  if export_boundary:
    outline_output_path = get_unique_filename(output_dir, 'outline', '.avi')
    outline_out = cv2.VideoWriter(outline_output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

  # Rewind the video to the start
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  # Calculate the total number of frames in the video
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Configure the progress bar
  progress_bar['maximum'] = total_frames
  progress_bar['value'] = 0

  # Use the K-means algorithm initialized during the preview
  for frame_num in range(total_frames):
    ret, frame = cap.read()
    if not ret:
      break

    # Resize the image
    image = cv2.resize(frame, (900, 900))
    image_np = np.array(image)

    # Convert the image to a 2D matrix
    if image_np.shape[2] == 4:
      pixels = image_np.reshape(-1, 4)  # RGBA image
    else:
      pixels = image_np.reshape(-1, 3)  # RGB image

    # Predict the labels using the initialized K-means
    labels = kmeans.predict(pixels)
    centroids = kmeans.cluster_centers_

    # Create a new image with cluster colors
    segmented_img = centroids[labels].reshape(image_np.shape).astype(np.uint8)

    # Select the cluster to draw the boundary
    mask = (labels == selected_cluster)
    masked_labels = np.where(mask.reshape(image_np.shape[:2]), labels.reshape(image_np.shape[:2]), -1)

    # Mark the boundaries of the selected cluster
    boundary_img = mark_boundaries(np.zeros_like(segmented_img), masked_labels, color=(0, 1, 0))

    # Convert boundary_img to a format that cv2 can use
    boundary_img = (boundary_img * 255).astype(np.uint8)

    # Resize to match the original size
    boundary_img = cv2.resize(boundary_img, (int(cap.get(3)), int(cap.get(4))))

    # Overlay the boundary markings on the original image
    frame_with_boundaries = cv2.addWeighted(frame, 1, boundary_img, 1, 0)

    # Write the frames to the videos
    if export_original:
      out.write(frame_with_boundaries)
    if export_boundary:
      outline_out.write(boundary_img)

    # Update progress bar
    progress_bar['value'] = frame_num + 1
    root.update_idletasks()

  # Release video objects
  cap.release()
  if export_original:
    out.release()
  if export_boundary:
    outline_out.release()
  cv2.destroyAllWindows()

  messagebox.showinfo("Info", "Video processing completed and saved.")

def select_video():
  global cap, preview_frame, kmeans
  video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
  if video_path:
    cap = cv2.VideoCapture(video_path)
    ret, preview_frame = cap.read()
    if not ret:
      messagebox.showerror("Error", "Failed to read video")
      return
    kmeans = None
    update_preview()

# Initial parameters for k-means
num_colors = 4
selected_cluster = 0
preview_frame = None

# Initialize the K-means algorithm globally
kmeans = None

# Create a Tkinter window for parameter adjustment
root = tk.Tk()
root.title("K-means Parameter Adjustment")

# Add a button to select the video file
btn_select_video = tk.Button(root, text="Select Video", command=select_video)
btn_select_video.pack(pady=10)

# Create a frame for the preview image and parameters
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label to display the preview image
preview_label = tk.Label(left_frame)
preview_label.pack()

# Create a frame for the buttons and current values
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create buttons to adjust parameters and display current values
tk.Label(right_frame, text="Clusters").grid(row=0, column=0, pady=5)
num_colors_label = tk.Label(right_frame, text=str(num_colors))
num_colors_label.grid(row=0, column=1, pady=5)

btn_increase_colors = tk.Button(right_frame, text="+", command=increase_colors)
btn_increase_colors.grid(row=1, column=0, pady=5)
btn_decrease_colors = tk.Button(right_frame, text="-", command=decrease_colors)
btn_decrease_colors.grid(row=1, column=1, pady=5)

tk.Label(right_frame, text="Boundary").grid(row=2, column=0, pady=5)
selected_cluster_label = tk.Label(right_frame, text=str(selected_cluster))
selected_cluster_label.grid(row=2, column=1, pady=5)

btn_increase_cluster = tk.Button(right_frame, text="+", command=increase_cluster)
btn_increase_cluster.grid(row=3, column=0, pady=5)
btn_decrease_cluster = tk.Button(right_frame, text="-", command=decrease_cluster)
btn_decrease_cluster.grid(row=3, column=1, pady=5)

# Create checkboxes to select which videos to export
export_original_var = tk.IntVar(value=1)
export_boundary_var = tk.IntVar(value=1)

chk_export_original = tk.Checkbutton(right_frame, text="Export Original Video", variable=export_original_var)
chk_export_original.grid(row=4, column=0, columnspan=2, pady=5)

chk_export_boundary = tk.Checkbutton(right_frame, text="Export Boundary Video", variable=export_boundary_var)
chk_export_boundary.grid(row=5, column=0, columnspan=2, pady=5)

btn_start_processing = tk.Button(right_frame, text="Export Video", command=start_processing)
btn_start_processing.grid(row=6, column=0, columnspan=2, pady=5)

btn_exit = tk.Button(right_frame, text="Exit", command=root.destroy)
btn_exit.grid(row=7, column=0, columnspan=2, pady=5)

# Add a progress bar
progress_bar = ttk.Progressbar(root, orient='horizontal', mode='determinate')
progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

root.mainloop()
