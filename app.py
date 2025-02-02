import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
import pandas as pd
from tkinter import Tk, Label, Entry, Button, filedialog

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    date TEXT,
                    time TEXT
                )''')
    conn.commit()
    conn.close()

# Save attendance to database
def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    c.execute('INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)', (name, date, time))
    conn.commit()
    conn.close()

# Export attendance to Excel
def export_to_excel():
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query('SELECT * FROM attendance', conn)
    df.to_excel('attendance_report.xlsx', index=False)
    conn.close()

# Add new user with photo
def add_user(name, photo_path):
    known_face_encodings = []
    known_face_names = []

    image = face_recognition.load_image_file(photo_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

    np.save('encodings.npy', known_face_encodings)
    np.save('names.npy', known_face_names)

# GUI for adding new user
def browse_image():
    file_path = filedialog.askopenfilename()
    return file_path

def submit_user():
    name = name_entry.get()
    photo_path = browse_image()
    if name and photo_path:
        add_user(name, photo_path)
        status_label.config(text='User added successfully!')
    else:
        status_label.config(text='Please enter a name and select a photo.')

# Face recognition for attendance
def recognize_and_mark_attendance():
    known_face_encodings = np.load('encodings.npy')
    known_face_names = np.load('names.npy')

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            mark_attendance(name)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# GUI Setup
root = Tk()
root.title('Smart Attendance System')

Label(root, text='Enter Name:').grid(row=0, column=0)
name_entry = Entry(root)
name_entry.grid(row=0, column=1)

Button(root, text='Add User', command=submit_user).grid(row=1, column=0, columnspan=2)
Button(root, text='Start Attendance', command=recognize_and_mark_attendance).grid(row=2, column=0, columnspan=2)
Button(root, text='Export to Excel', command=export_to_excel).grid(row=3, column=0, columnspan=2)

status_label = Label(root, text='')
status_label.grid(row=4, column=0, columnspan=2)

init_db()
root.mainloop()
