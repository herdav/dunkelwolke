# loadSunVideos
# Created 2024-07-01 by David Herren

import os
import requests
from datetime import datetime, timedelta
from tkinter import Tk, filedialog, Toplevel, Label, Button, OptionMenu, StringVar, IntVar, Checkbutton
from tkcalendar import Calendar
from tkinter.ttk import Progressbar
from moviepy.editor import VideoFileClip, concatenate_videoclips

def download_file(url, save_path):
  try:
    response = requests.get(url)
    if response.status_code == 200:
      with open(save_path, 'wb') as file:
        file.write(response.content)
      return True
    else:
      return False
  except Exception as e:
    print(f"Error downloading {url}: {e}")
    return False

def select_dates_and_type():
  def on_select():
    start_date = cal_start.selection_get()
    end_date = cal_end.selection_get()
    date_range.set(f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}")
    selected_file_type.set(file_type.get())
    concatenate_videos.set(concat_videos.get())
    selection_window.destroy()
    show_progress_window()

  selection_window = Toplevel(root)
  selection_window.title("Please select date range and file type")
  
  Label(selection_window, text="Start date:").grid(row=0, column=0, padx=10, pady=5)
  cal_start = Calendar(selection_window, selectmode='day')
  cal_start.grid(row=0, column=1, padx=10, pady=5)
  
  Label(selection_window, text="End date:").grid(row=1, column=0, padx=10, pady=5)
  cal_end = Calendar(selection_window, selectmode='day')
  cal_end.grid(row=1, column=1, padx=10, pady=5)
  
  Label(selection_window, text="File type:").grid(row=2, column=0, padx=10, pady=5)
  file_types = sorted(["0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600", "1700", "HMIIC", "HMIBC", "HMID", "HMII", "HMIIF"])
  file_type = StringVar(selection_window)
  file_type.set(file_types[0])
  file_type_menu = OptionMenu(selection_window, file_type, *file_types)
  file_type_menu.grid(row=2, column=1, padx=10, pady=5)
  
  concat_videos = IntVar()
  Checkbutton(selection_window, text="Concatenate videos", variable=concat_videos).grid(row=3, columnspan=2, pady=5)
  
  Button(selection_window, text="Confirm", command=on_select).grid(row=4, columnspan=2, pady=20)

def concatenate_downloaded_videos_by_month(video_files, file_type, progress_window, progress_bar, progress_label):
  # Group videos by year and month
  video_groups = {}
  for file in video_files:
    year_month = os.path.basename(file)[:6]
    if year_month not in video_groups:
      video_groups[year_month] = []
    video_groups[year_month].append(file)

  total_groups = len(video_groups)
  current_group = 0

  for year_month, files in video_groups.items():
    clips = []
    total_clips = len(files)
    for idx, file in enumerate(files):
      clips.append(VideoFileClip(file))
      progress_label.config(text=f"Loading video {idx + 1}/{total_clips} for concatenation of {year_month}...")
      progress_bar['value'] = (idx + 1) / total_clips * 100
      progress_window.update_idletasks()

    final_clip = concatenate_videoclips(clips, method="compose")
    output_video_path = os.path.join(os.path.dirname(files[0]), f"merge_{year_month}_1024_{file_type}.mp4")
    progress_label.config(text=f"Saving concatenated video for {year_month}...")
    final_clip.write_videofile(output_video_path)
    progress_label.config(text=f"Videos for {year_month} concatenated successfully!")
    current_group += 1
    progress_bar['value'] = current_group / total_groups * 100
    progress_window.update_idletasks()

def show_progress_window():
  progress_window = Toplevel(root)
  progress_window.title("Download Progress")

  progress_label = Label(progress_window, text="Starting download...")
  progress_label.pack(pady=10)

  progress_bar = Progressbar(progress_window, orient='horizontal', length=400, mode='determinate')
  progress_bar.pack(pady=10)

  start_date_str, end_date_str = date_range.get().split(',')
  start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
  end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
  current_date = start_date
  total_days = (end_date - start_date).days + 1
  progress_bar['maximum'] = total_days

  downloaded_files = []
  failed_files = []

  base_url = "https://sdo.gsfc.nasa.gov/assets/img/dailymov/"

  day_count = 0
  while current_date <= end_date:
    year_month = current_date.strftime("%Y/%m")
    date_str = current_date.strftime("%Y%m%d")
    file_name = f"{date_str}_1024_{selected_file_type.get()}.mp4"
    url = f"{base_url}{year_month}/{current_date.day:02d}/{file_name}"
    save_path = os.path.join(save_directory, year_month, file_name)

    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))

    if os.path.exists(save_path):
      print(f"File {file_name} already exists, skipping download.")
      downloaded_files.append(save_path)
    else:
      progress_label.config(text=f"Downloading: {file_name}")
      progress_window.update_idletasks()
      
      if download_file(url, save_path):
        downloaded_files.append(save_path)
      else:
        failed_files.append(file_name)
    
    current_date += timedelta(days=1)
    day_count += 1
    progress_bar['value'] = day_count
    progress_window.update_idletasks()

  if failed_files:
    progress_label.config(text="Download complete with some errors. Check log file.")
  else:
    progress_label.config(text="Download complete successfully!")
  
  progress_window.update_idletasks()
  
  # Concatenate videos if the checkbox was selected
  if concatenate_videos.get() == 1 and downloaded_files:
    progress_bar['value'] = 0  # Reset progress bar for concatenation
    concatenate_downloaded_videos_by_month(downloaded_files, selected_file_type.get(), progress_window, progress_bar, progress_label)

  # Close the progress window after download completion
  def close_progress_window():
    progress_window.destroy()
    select_dates_and_type()  # Allow the user to start a new download

  Button(progress_window, text="Close", command=close_progress_window).pack(pady=20)
  
  print("Downloaded files:")
  for file in downloaded_files:
    print(file)

  print("\nFailed files:")
  for file in failed_files:
    print(file)

  # Index log file
  log_file_base = os.path.join(save_directory, "download_log.txt")
  log_file = log_file_base
  i = 1
  while os.path.exists(log_file):
    log_file = f"{log_file_base[:-4]}_{i}.txt"
    i += 1

  with open(log_file, 'w') as log:
    log.write("Downloaded files:\n")
    for file in downloaded_files:
      log.write(f"{file}\n")
    
    log.write("\nFailed files:\n")
    for file in failed_files:
      log.write(f"{file}\n")

  print(f"\nLog file created: {log_file}")

root = Tk()
root.withdraw()

# Select save directory
save_directory = filedialog.askdirectory(title="Please select the save directory")
if not save_directory:
  print("No save directory selected. Exiting.")
  exit()

# Select date range and file type
date_range = StringVar()
selected_file_type = StringVar()
concatenate_videos = IntVar()
select_dates_and_type()
root.wait_window()

# Start the Tkinter main loop
root.mainloop()
