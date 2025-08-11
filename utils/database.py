import os
import time
import sqlite3
from config import CONFIG



def initialize_database():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(CONFIG['database'])
    print('Connected to database!')
    cursor = conn.cursor()
    # Create sus_person table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sus_person (
            path TEXT,
            time TEXT
        )
    ''')
    # Create item_abandon table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS item_abandon (
            path TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()


def add_record(table_name, path, time):
    """Add a new record to the specified table."""
    conn = sqlite3.connect(CONFIG['database'])
    cursor = conn.cursor()
    
    cursor.execute(f'''
        INSERT INTO {table_name} (path, time)
        VALUES (?, ?)
    ''', (path, time))
    
    conn.commit()
    conn.close()


def update_record(table_name, old_path, new_path, new_time):
    """Update an existing record in the specified table."""
    conn = sqlite3.connect(CONFIG['database'])
    cursor = conn.cursor()
    
    cursor.execute(f'''
        UPDATE {table_name}
        SET path = ?, time = ?
        WHERE path = ?
    ''', (new_path, new_time, old_path))
    
    conn.commit()
    conn.close()


def delete_record(table_name, path):
    """Delete a record from the specified table."""
    conn = sqlite3.connect(CONFIG['database'])
    cursor = conn.cursor()
    
    cursor.execute(f'''
        DELETE FROM {table_name}
        WHERE path = ?
    ''', (path,))
    
    conn.commit()
    conn.close()