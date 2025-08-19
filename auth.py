import sqlite3
import bcrypt
import streamlit as st
import os

DB_PATH = os.path.join('data', 'users.db')

def create_db():
    """Initialize database and tables"""
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, 
                      password TEXT, 
                      role TEXT DEFAULT 'user')''')
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        conn.close()

def create_account(username, password, role='user'):
    """Create new user account"""
    create_db()
    if not username or not password:
        st.error("Username and password required")
        return False

    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                 (username, hashed.decode(), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists")
        return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False
    finally:
        conn.close()

def authenticate(username, password):
    """Verify user credentials"""
    create_db()
    if not username or not password:
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password, role FROM users WHERE username=?", (username,))
        result = c.fetchone()
        
        if result and bcrypt.checkpw(password.encode(), result[0].encode()):
            st.session_state.user_role = result[1]
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False
    finally:
        conn.close()