import xml.etree.ElementTree as ET
import psycopg2
from psycopg2 import extras
import os
from datetime import datetime, timezone
from tqdm import tqdm
import uuid

# Database connection parameters
DB_PARAMS = {
    "dbname": "bold_tracking",
    "user": "admin",
    "password": "CrazySecure",
    "host": "kaspersvendsen.dk",
    "port": "5432"
}

def truncate_tables(cur):
    """Truncates the relevant tables in the database."""
    print("Truncating tables...")
    cur.execute("TRUNCATE TABLE tracking_data, ball_tracking, players CASCADE;")
    print("Tables truncated!")

def parse_and_insert(xml_file, chunk_size=5000):
    """Parses the XML file and inserts data into the database."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Truncate tables before inserting new data
    truncate_tables(cur)
    conn.commit()

    # Count total frames for progress bar
    total_frames = sum(1 for _ in root.findall(".//frame"))

    player_buffer = []
    tracking_buffer = []
    ball_buffer = []

    # Create temporary tables
    print("Creating temporary tables...")
    cur.execute("""
        CREATE TEMPORARY TABLE temp_players (
            id UUID PRIMARY KEY,
            opta_id INTEGER,
            team TEXT CHECK (team IN ('home', 'away')),
            number INTEGER
        );

        CREATE TEMPORARY TABLE temp_tracking_data (
            time TIMESTAMPTZ NOT NULL,
            period INTEGER,
            wall_clock BIGINT,
            live BOOLEAN,
            possession TEXT CHECK (possession IN ('home', 'away')),
            player_id UUID,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            spd FLOAT,
            dist FLOAT
        );

        CREATE TEMPORARY TABLE temp_ball_tracking (
            time TIMESTAMPTZ NOT NULL,
            period INTEGER,
            wall_clock BIGINT,
            live BOOLEAN,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            spd FLOAT,
            dist FLOAT
        );
    """)
    conn.commit()
    print("Temporary tables created!")

    # Process frames in the XML
    processed_players = set()

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        for period in root.findall("period"):
            period_num = int(period.get("number"))

            for frame in period.findall("frame"):
                time = datetime.now(timezone.utc)
                wall_clock = int(frame.get("wall_clock"))
                live = frame.get("live") == "True"
                possession = frame.get("possession")
                if possession:
                    possession = possession.lower()

                for player in frame.findall("player"):
                    player_id = str(uuid.UUID(player.get("id")))
                    number = int(player.get("num"))
                    opta_id = int(player.get("opta_id"))
                    loc_str = player.get("loc")
                    spd = float(player.get("spd"))
                    dist = float(player.get("dist"))

                    if loc_str is None or "null" in loc_str.lower():
                        loc = (None, None, None)
                    else:
                        loc = eval(loc_str)

                    if player_id not in processed_players:
                        player_buffer.append((player_id, opta_id, 'home', number))
                        processed_players.add(player_id)

                    tracking_buffer.append((time, period_num, wall_clock, live, possession,
                                            player_id, loc[0], loc[1], loc[2], spd, dist))

                ball = frame.find("ball")
                if ball is not None:
                    ball_loc_str = ball.get("loc")
                    ball_spd = float(ball.get("spd")) if ball.get("spd") else None
                    ball_dist = float(ball.get("dist")) if ball.get("dist") else None

                    if ball_loc_str is None or "null" in ball_loc_str.lower():
                        ball_loc = (None, None, None)
                    else:
                        ball_loc = eval(ball_loc_str)

                    ball_buffer.append((time, period_num, wall_clock, live,
                                        ball_loc[0], ball_loc[1], ball_loc[2],
                                        ball_spd, ball_dist))

                if len(tracking_buffer) >= chunk_size:
                    insert_chunks(cur, player_buffer, tracking_buffer, ball_buffer)
                    player_buffer = []
                    tracking_buffer = []
                    ball_buffer = []
                    conn.commit()

                pbar.update(1)

        # Insert remaining data
        if tracking_buffer:
            insert_chunks(cur, player_buffer, tracking_buffer, ball_buffer)
            conn.commit()

    # Move data from temporary tables to permanent tables
    print("Moving data to permanent tables...")
    cur.execute("""
        INSERT INTO players 
        SELECT * FROM temp_players 
        ON CONFLICT (id) DO NOTHING;

        INSERT INTO tracking_data (time, period, wall_clock, live, possession, player_id, x, y, z, spd, dist)
        SELECT time, period, wall_clock, live, possession, player_id, x, y, z, spd, dist 
        FROM temp_tracking_data;

        INSERT INTO ball_tracking (time, period, wall_clock, live, x, y, z, spd, dist)
        SELECT time, period, wall_clock, live, x, y, z, spd, dist 
        FROM temp_ball_tracking;
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Data from {xml_file} inserted successfully!")

def insert_chunks(cur, player_buffer, tracking_buffer, ball_buffer):
    """Inserts data into temporary tables in chunks."""
    if player_buffer:
        extras.execute_values(
            cur,
            """
            INSERT INTO temp_players (id, opta_id, team, number)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            player_buffer
        )

    if tracking_buffer:
        extras.execute_values(
            cur,
            """
            INSERT INTO temp_tracking_data (
                time, period, wall_clock, live, possession, 
                player_id, x, y, z, spd, dist
            )
            VALUES %s
            """,
            tracking_buffer
        )

    if ball_buffer:
        extras.execute_values(
            cur,
            """
            INSERT INTO temp_ball_tracking (
                time, period, wall_clock, live, 
                x, y, z, spd, dist
            )
            VALUES %s
            """,
            ball_buffer
        )

if __name__ == "__main__":
    xml_file = "matches/1/tracking-produced.xml"
    if os.path.exists(xml_file):
        parse_and_insert(xml_file)
    else:
        print(f"XML file {xml_file} not found!")
