import cv2
import face_recognition
import numpy as np
import os
os.environ['DISPLAY'] = ':0'  # helps in some Mac environments
import pandas as pd
from datetime import datetime
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import simpledialog

# ==== Load Known Faces ====
path = 'known_faces'
images = []
classNames = []
print("Starting the Face Recognition GUI...")
for filename in os.listdir(path):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img = cv2.imread(f'{path}/{filename}')
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])
def load_known_faces():
    path = 'known_faces'
    images = []
    classNames = []

    for file in os.listdir(path):
        curImg = cv2.imread(os.path.join(path, file))
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(file)[0])

    return images, classNames
def backup_and_reset_attendance(file_path):
    try:
        if not os.path.exists(file_path):
            messagebox.showwarning("Warning", "⚠️ Attendance.csv does not exist.")
            return

        df = pd.read_csv(file_path)

        # Generate backup filename with date and session
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        session = "FN" if now.hour < 13 else "AN"
        backup_folder = os.path.join(os.path.dirname(file_path), "Attendance_Backups")
        os.makedirs(backup_folder, exist_ok=True)
        backup_filename = f"Attendance_Backup_{date_str}_{session}.csv"
        backup_path = os.path.join(backup_folder, backup_filename)

        # Save backup
        df.to_csv(backup_path, index=False)

        # Reset original file (keep only Name column if present)
        if "Name" in df.columns:
            df = df[["Name"]]
        else:
            df = pd.DataFrame(columns=["Name"])
        df.to_csv(file_path, index=False)

        messagebox.showinfo("Success", f"✅ Backup created:\n{backup_filename}\n\n🧹 Attendance reset.")

    except Exception as e:
        messagebox.showerror("Error", f"Error during reset:\n{str(e)}")
def findEncodings(images):
    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(rgb_img)[0])
    return encodeList

encodeListKnown = findEncodings(images)

# ==== Attendance Logic ====
last_marked = {}
last_warned = {}
MARK_COOLDOWN = 10
WARN_COOLDOWN = 5

def markAttendance(name, file_path="Attendance.csv"):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    session = 'FN' if now.hour < 13 else 'AN'
    column_name = f"{date_str} {session}"

    # Load CSV or create new
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name"])

    # Add new student if not present
    if name not in df["Name"].values:
        df = pd.concat([df, pd.DataFrame([{"Name": name}])], ignore_index=True)

    # Ensure the session column exists
    if column_name not in df.columns:
        df[column_name] = ""

    # Find row index
    idx = df[df["Name"] == name].index[0]

    # Mark attendance only if not already marked
    if df.at[idx, column_name] != "✅":
        df.at[idx, column_name] = "✅"
        print(f"✅ Marked {name} present for {column_name}")
    else:
        print(f"⏳ {name} already marked for {column_name}")

    # Save the updated sheet
    df.to_csv(file_path, index=False)

# ==== GUI with Tkinter ====

class AttendanceApp:
    def __init__(self, root):
        self.current_name = None
        self.last_marked = {}
        self.last_warned = {}
        self.MARK_COOLDOWN = 10
        self.WARN_COOLDOWN = 5
        self.root = root
        self.root.title("Face Recognition Attendance System")

        try:
            logo_img = Image.open("anurag_logo.png")
            logo_img = logo_img.resize((120, 120))
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(root, image=self.logo)
            logo_label.pack(pady=5)
        except:
            pass  # Skip if logo image not found
        self.root.geometry("900x700")
        self.video_running = False
        self.images, self.classNames = load_known_faces()
        self.encodeListKnown = findEncodings(self.images)

        self.label = tk.Label(root, text="Welcome, Abhilash!", font=("Helvetica", 20, "bold"))
        self.label.pack(pady=10)

        self.status = tk.Label(root, text="Welcome, Abhilash!", font=("Helvetica", 14),bg="#121212", fg="#00E676")  # Green text on dark
        self.status.pack(pady=5)

        self.start_btn = tk.Button(root, text="🎥 Start Attendance", font=("Helvetica", 14), bg="#1E88E5", fg="black", activebackground="#1565C0", activeforeground="black",
                                   command=self.start_camera)
        self.start_btn.pack(pady=10)

        self.mark_btn = tk.Button(root, text="📝 Mark Attendance", font=("Helvetica", 14),bg="#43A047", fg="black", activebackground="#2E7D32", activeforeground="black",
                                  command=self.mark_button_click)
        self.mark_btn.pack(pady=10)

        self.view_btn = tk.Button(root, text="📄 View Attendance", font=("Helvetica", 12), bg="#6A1B9A", fg="black", activebackground="#4A148C", activeforeground="black",
                                  command=self.show_attendance)
        self.view_btn.pack(pady=5)
        self.add_face_btn = tk.Button(root, text="➕ Add New Face", font=("Helvetica", 12),
                              bg="#FF5722", fg="white", activebackground="#E64A19",
                              command=self.add_new_face)
        self.add_face_btn.pack(pady=5)
        reset_btn = tk.Button(self.root, text="🔁 Backup & Reset Attendance", command=lambda: backup_and_reset_attendance("Attendance.csv"), bg="#333", fg="white", font=("Helvetica", 12))
        reset_btn.pack(pady=10)
        self.clock_label = tk.Label(root, text="", font=("Helvetica", 14),
                                    bg="#121212", fg="#FFC107")
        self.clock_label.pack()
        self.update_clock()

        self.canvas = tk.Label(root)
        self.canvas.pack()
    def mark_button_click(self):
        name = self.current_name
        current_time = time.time()

        if name:
            if name not in self.last_marked or (current_time - self.last_marked[name]) > self.MARK_COOLDOWN:
                markAttendance(name)
                self.last_marked[name] = current_time
                self.status.config(text=f"✅ {name} marked present", fg="green")
            else:
                if name not in self.last_warned or (current_time - self.last_warned[name]) > self.WARN_COOLDOWN:
                    self.status.config(text=f"⏳ Wait to re-mark {name}", fg="orange")
                    self.last_warned[name] = current_time
        else:
            self.status.config(text="❗ No face detected", fg="red")
    def add_new_face(self):
        name = simpledialog.askstring("Input", "Enter name of the person:")
        if not name:
            return

        folder = "Known_faces"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{name}.jpg")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Camera Error", "🔴 Could not access the camera.")
            return

        cv2.namedWindow("Add New Face", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Add New Face", 800, 600)

        captured = False

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press 's' to Save or 'q' to Quit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Add New Face", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(file_path, frame)
                messagebox.showinfo("Saved", f"✅ Face saved as {name}.jpg")
                captured = True
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
   
        if not captured:
            messagebox.showinfo("Cancelled", "Face not saved.")
        # Reload known faces
        self.images, self.classNames = load_known_faces()
        self.encodeListKnown = findEncodings(self.images)
        self.status.config(text="✅ Reloaded faces", fg="#00E676")
    def reload_known_faces(self):
        path = 'known_faces'
        self.classNames = []
        images = []

        for cl in os.listdir(path):
            img = cv2.imread(f'{path}/{cl}')
            if img is not None:
                images.append(img)
                self.classNames.append(os.path.splitext(cl)[0])
    def findEncodings(images):
            encodeList = []
            for img in images:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encode = face_recognition.face_encodings(img_rgb)
                    if encode:
                        encodeList.append(encode[0])
                except:
                    continue
            return encodeList

            self.known_face_encodings = findEncodings(images)
            self.status.config(text="🔄 Known faces reloaded", fg="#00BCD4")
    def show_attendance(self):
        try:
            df = pd.read_csv('Attendance.csv')
            top = tk.Toplevel(self.root)
            top.title("Attendance Records")

            text = tk.Text(top, wrap='none')
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(top, command=text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.config(yscrollcommand=scrollbar.set)

            text.insert(tk.END, df.to_string(index=False))
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't load Attendance.csv:\n{e}")
    def update_clock(self):
        now = datetime.now().strftime('%H:%M:%S')
        self.clock_label.config(text=f"🕒 {now}")
        self.root.after(1000, self.update_clock)
    def start_camera(self):
        if not self.video_running:
            self.video_running = True
            threading.Thread(target=self.run_camera).start()
        self.start_btn.config(bg="#1E88E5", fg="white", activebackground="#1565C0", activeforeground="white")
        self.mark_btn.config(bg="#43A047", fg="white", activebackground="#2E7D32", activeforeground="white")
        self.view_btn.config(bg="#6A1B9A", fg="white", activebackground="#4A148C", activeforeground="white")
        self.status.config(bg="#121212", fg="#00E676")
        self.clock_label.config(bg="#121212", fg="#FFC107")
    def run_camera(self):
        cap = cv2.VideoCapture(0)

        while self.video_running:
            ret, frame = cap.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(rgb_small_frame)
            encodesCurFrame = face_recognition.face_encodings(rgb_small_frame, facesCurFrame)

            current_time = time.time()

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)

                if len(faceDis) == 0:
                    continue

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    self.current_name = name  # Just store for now

                    y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Display the frame in Tkinter
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        cap.release()

# ==== Run App ====
if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#121212")  # Dark background
    app = AttendanceApp(root)
    root.mainloop()
