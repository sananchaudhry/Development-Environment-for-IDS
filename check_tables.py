import sqlite3

# Connect to database
conn = sqlite3.connect("ids_ml.db")
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

# Print contents of each table
for table in tables:
    print(f"\nTable: {table}")
    cursor.execute(f"SELECT * FROM {table};")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

conn.close()

