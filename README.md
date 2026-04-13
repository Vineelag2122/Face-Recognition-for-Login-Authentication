# Face Recognition Login Authentication

A Flask project for face-based user authentication using:

- Flask (backend)
- OpenCV (image handling)
- face_recognition / dlib (face encoding + matching)
- Bootstrap (frontend)
- SQLite (user data storage)

## Features

- Register user with live camera stream (auto-captured on submit)
- Store only face encoding in SQLite (no image file storage)
- Login by live camera stream (auto-captured on submit)
- Compare input face with stored encoding
- Return login success or access denied

## Project Structure

```text
c:\AI
|-- app.py
|-- requirements.txt
|-- users.db                # created automatically at runtime
|-- static/
|   `-- styles.css
`-- templates/
    |-- base.html
    |-- index.html
    |-- register.html
    |-- login.html
    `-- login_result.html
```

## Setup

1. Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If `dlib` fails to build on Windows, install **Visual Studio Build Tools** with the **Desktop development with C++** workload, then run the install again. You may also need:

- Windows 10/11 SDK
- CMake
- A clean terminal after installation

3. Run app:

```powershell
python app.py
```

4. Open browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. Register:
- Enter username
- Allow camera access and submit while face is visible
- System captures current live frame, extracts 128-d encoding, and saves it in DB

2. Login:
- Enter username
- Allow camera access and submit while face is visible
- System compares face distance against threshold (`0.5`)
- If below threshold: success

## Camera Note

- The browser must allow camera access for registration and login.
- This works best on `http://127.0.0.1:5000` or `localhost` during development.
- If the camera does not start, check browser permissions and refresh the page.

## Important Notes

- Use clear, front-facing images for better accuracy.
- `face_recognition` requires `dlib`, which may take time to install on Windows.
- On Windows, `dlib` often fails unless Visual C++ build tools are installed.
- If you want the easiest setup, use Python 3.10 or 3.11 in a fresh virtual environment.
- Change `SECRET_KEY` in `app.py` before production.
- This is a learning/demo project; for production, add proper auth/session controls, encryption, and anti-spoofing.

# model
Face embedding model: dlib’s 128-dimensional face descriptor (ResNet-based) via face_recognition.face_encodings.
Face detection mode here: default face_recognition behavior (HOG-based detector, since no CNN model option is set in code).
The MediaPipe model file in models/ is for phone-object anti-spoof checks, not for face recognition.