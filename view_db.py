import sqlite3
import pickle
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "users.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# See all users
cursor.execute("SELECT id, username, created_at FROM users")
rows = cursor.fetchall()

if not rows:
    print("No users in database")
else:
    print(f"Found {len(rows)} user(s):")
    print("-" * 60)
    for row in rows:
        print(f"ID: {row[0]}, Username: {row[1]}, Created: {row[2]}")

conn.close()
