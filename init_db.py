import sqlite3
conn = sqlite3.connect("ids_ml.db")
with open("schema.sql") as f:
    conn.executescript(f.read())
conn.commit()
conn.close()
print("Database initialized.")
