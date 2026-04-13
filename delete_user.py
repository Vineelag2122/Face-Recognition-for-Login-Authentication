import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "users.db"

def delete_user(username: str) -> bool:
    """Delete a user by username"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    
    if cursor.rowcount > 0:
        print(f"✓ Deleted user: {username}")
        conn.close()
        return True
    else:
        print(f"✗ User not found: {username}")
        conn.close()
        return False

def delete_user_by_id(user_id: int) -> bool:
    """Delete a user by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    
    if cursor.rowcount > 0:
        print(f"✓ Deleted user with ID: {user_id}")
        conn.close()
        return True
    else:
        print(f"✗ User not found with ID: {user_id}")
        conn.close()
        return False

# Example usage:
if __name__ == "__main__":
    # Delete all users
    delete_user("Vineela")
    delete_user("Ujjwal")
    #delete_user("Vineela G")
