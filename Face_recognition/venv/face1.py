import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# ==== SETTINGS ====
DB_PATH = "face_database"           # folder with passport images
ATTENDANCE_FILE = "attendance.csv"  # CSV file to save attendance

# ==== CREATE ATTENDANCE FILE IF NOT EXISTS ====
try:
    df = pd.read_csv(ATTENDANCE_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# ==== HELPER FUNCTION TO MARK ATTENDANCE ====
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)

    # Avoid duplicate marking for the same day
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[INFO] Attendance marked for {name} at {time}")
    else:
        print(f"[INFO] {name} already marked today.")

# ==== START WEBCAM ====
cap = cv2.VideoCapture(0)
print("[INFO] Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze only recognition (disable age/emotion)
        objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",  # accurate face detector
            enforce_detection=False
        )

        for obj in objs:
            x, y, w, h = obj["facial_area"].values()
            face_img = frame[y:y+h, x:x+w]

            # Verify with database
            result = DeepFace.find(
                img_path=face_img,
                db_path=DB_PATH,
                model_name="Facenet",        # lightweight & accurate
                enforce_detection=False
            )

            if not result.empty:
                person_name = result.iloc[0]["identity"].split("\\")[-1].split(".")[0]
                mark_attendance(person_name)

                # Draw box & name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        print("[ERROR]", e)

    cv2.imshow("University Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance system stopped.")
