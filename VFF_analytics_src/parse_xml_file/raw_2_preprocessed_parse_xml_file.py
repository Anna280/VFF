import pandas as pd
import xml.etree.ElementTree as ET
import ast
from tqdm import tqdm

class ParseTrackingXML:
    
    def __init__(self, path):
        try:
            print("loading the xml file...")
            self.data = ET.parse(path)
            print("xml file loaded!")
        except Exception as e:
            self.data = None
            print(f"Error reading file: {e}")
            
    def extract_player_data(self, extend_field_x = 53, extend_field_y = 34):
        root = self.data.getroot()
        unique_id = str("viborg")
        players_rows = []   
    
        for period in root.findall("period"):
            period_number = int(period.get('number'))
            
            frames = period.findall("frame")
            for frame in tqdm(frames, desc="Processing frames (players)", leave=False):
                timestamp = float(frame.get('time'))

                for player in frame.findall('player'):
                    player_num = int(player.get('num'))
                    loc = ast.literal_eval(player.get('loc')) 
                    spd = float(player.get('spd')) 
                    player_id = str(player.get("id"))

                    players_rows.append({
                        'unique_id': unique_id,  
                        'period': period_number,
                        'time': timestamp,
                        'player_id':player_id,
                        'player_num': player_num,
                        'x': round(loc[0] + extend_field_x, 2), 
                        'y': round(loc[1] + extend_field_y, 2),
                        'z': loc[2],
                        'spd': spd
                    })
            data_players = pd.DataFrame(players_rows)
            data_players = data_players.sort_values(["period", "time"]).reset_index(drop=True)
        return data_players
    
    def extract_ball_data(self, extend_field_x = 53, extend_field_y = 34):

        root = self.data.getroot()
        ball_rows = []
        for period_elem in root.findall("period"):
            period_number = period_elem.get("number")
            frames = period_elem.findall("frame")
            for frame in tqdm(frames, desc="Processing frames (ball)", leave=False):
                time_val = float(frame.get("time"))
                ball_elem = frame.find("ball")
                loc_str = ball_elem.get("loc").strip("[]")
                parts = loc_str.split(",")
                try:
                    ball_x = float(parts[0].strip())
                    ball_y = float(parts[1].strip())
                except ValueError:
                    continue

                ball_rows.append({
                    "time": time_val, 
                    "ball_x": ball_x + extend_field_x,  
                    "ball_y": ball_y + extend_field_y, 
                    "period": int(period_number)
                })
        
        ball_dataframe = pd.DataFrame(ball_rows)
        ball_dataframe = ball_dataframe.sort_values(["period", "time"]).reset_index(drop=True)
        return ball_dataframe
    
if __name__ == "__main__":
    date_of_match = "23-02-24"
    my_xml_parser = ParseTrackingXML("VFF_analytics_src/data/01_raw/tracking-produced.xml")
    players_dataframe = my_xml_parser.extract_player_data()
    ball_dataframe =  my_xml_parser.extract_ball_data()
    ball_dataframe.to_csv(f"VFF_analytics_src/data/02_preprocessed/viborg_ball_gps_{date_of_match}.csv")
    players_dataframe.to_csv(f"VFF_analytics_src/data/02_preprocessed/viborg_players_gps_{date_of_match}.csv")
    