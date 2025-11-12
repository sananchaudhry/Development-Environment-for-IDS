import sqlite3

# Create or overwrite database file
conn = sqlite3.connect("ids_ml.db")

# Read schema.sql and execute it
with open("schema.sql", "r") as f:
    conn.executescript(f.read())

conn.commit()
conn.close()

print("Database initialized successfully.")
