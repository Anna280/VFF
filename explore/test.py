import psycopg2

# Database connection parameters
DB_PARAMS = {
    "dbname": "bold_tracking",
    "user": "admin",
    "password": "CrazySecure",
    "host": "kaspersvendsen.dk",
    "port": "5432"
}

try:
    # Connect to the PostgreSQL database
    connection = psycopg2.connect(**DB_PARAMS)
    cursor = connection.cursor()

    # Query to get all tables and their schemas
    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
    AND table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name;
    """
    cursor.execute(query)

    # Fetch and print results
    tables = cursor.fetchall()
    print("Tables and their schemas:")
    for schema, table in tables:
        print(f"Schema: {schema}, Table: {table}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if connection:
        connection.close()
