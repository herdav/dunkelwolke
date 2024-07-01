# Created 2024-06-24 by David Herren

import os
import requests
from datetime import datetime
from tkinter import Tk, filedialog

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
    return False

root = Tk()
root.withdraw()
save_directory = filedialog.askdirectory(title="Bitte wählen Sie den Speicherort aus")

if not save_directory:
  print("Kein Speicherort ausgewählt. Das Programm wird beendet.")
  exit()

downloaded_files = []
failed_files = []

base_url = "https://sdo.gsfc.nasa.gov/assets/img/dailymov/2024/01/"
for day in range(1, 32):
  date_str = f"202401{day:02d}"
  # file_name = f"{date_str}_1024_HMIIC.mp4"
  file_name = f"{date_str}_1024_HMIBC.mp4"
  url = f"{base_url}{day:02d}/{file_name}"
  save_path = os.path.join(save_directory, file_name)
  
  if download_file(url, save_path):
    downloaded_files.append(file_name)
  else:
    failed_files.append(file_name)

print("Heruntergeladene Dateien:")
for file in downloaded_files:
  print(file)

print("\nFehlgeschlagene Dateien:")
for file in failed_files:
  print(file)

log_file = os.path.join(save_directory, "download_log.txt")
with open(log_file, 'w') as log:
  log.write("Heruntergeladene Dateien:\n")
  for file in downloaded_files:
    log.write(f"{file}\n")
  
  log.write("\nFehlgeschlagene Dateien:\n")
  for file in failed_files:
    log.write(f"{file}\n")

print(f"\nLog-Datei wurde erstellt: {log_file}")
