## 📘 Project Overview

This project provides two decision-support tools for **Viborg Fodsports Forening (Viborg FF)**:

1. **Pass Extraction Model**  
   Automatically detects pass events from GPS data with an **F1 score of 87–90%**.  
   Includes a **Streamlit-based web interface** for coaches to extract and review video clips of passes.

2. **Behavioral Decision Model**  
   A cognitive model that simulates how central defenders make passing decisions based on factors like:
   - Shot distance
   - Opponent-controlled space
   - Pass direction

## 🧠 Behavioral Decision Models

These models analyze player passing behavior using hierarchical Bayesian models. Posterior results from these models are evaluated in `evaluate_model.ipynb`.

---

## Pass Extraction Evaluation

The pass extraction component can be run using open-source data. For access to **Viborg FF-specific data**, please contact me directly.

If you have access to Viborg FF data, follow this pipeline:

1. Use `merge_and_filter_passes.py` to generate:
   - `preprocessed_data_for_decision_making.csv`
   - `test_for_streamlit.csv`

2. Place the annotation files in the correct folders (see below).
## 📦 Sample Data Note

The Metrica open-source files (e.g., Sample_Game_1_RawEventsData.csv) can be downloaded from:

➡️ https://github.com/metrica-sports/sample-data/tree/master/data

4. Run your chosen behavioral model script. Posterior results will be saved and used in the `evaluate_model.ipynb` notebook.

---
```bash
├── behavioral_models
│   ├── evlauate_model.ipynb
│   ├── hierarchical_weighted_triangle.py
│   ├── hierarhcical_bayes_changing_radius.py
│   ├── simple_hierarchical_bayes.py
│   └── utils
│       └── evaluation_utils.py
├── data
│   ├── 01_raw
│   │   ├── sample_match_1  
│   │   │   ├── Sample_Game_1_RawEventsData.csv
│   │   │   ├── Sample_Game_1_RawTrackingData_Away_Team.csv  
│   │   │   ├── Sample_Game_1_RawTrackingData_Home_Team.csv  
│   │   │   ├── labelled_match_ball_match.csv 
│   │   │   └── labelled_match_players_match.csv
│   │   ├── sample_match_2
│   │   │   ├── Sample_Game_2_RawEventsData.csv  
│   │   │   ├── Sample_Game_2_RawTrackingData_Away_Team.csv  
│   │   │   ├── Sample_Game_2_RawTrackingData_Home_Team.csv  
│   │   │   ├── labelled_match_ball_match2.csv
│   │   │   └── labelled_match_players_match2.csv
│   │   └── tracking-produced.xml   #Viborg FF data
│   ├── 02_preprocessed
│   │   ├── decision_model_data
│   │   │   ├── player_2.csv   #annotations 
│   │   │   ├── player_24.csv  #annotations 
│   │   ├── viborg_ball_gps_23-02-24.csv # preprocessed data generated throufh merge_and_filter_passes.py
│   │   └── viborg_players_gps_23-02-24.csv # preprocessed data generated throufh merge_and_filter_passes.py
│   └── 03_model_data
│       ├── preprocessed_data_for_decision_making.csv
│       └── test_for_streamit.csv # the data used for deploying the streamlit service
├── parse_xml_file
│   └── raw_2_preprocessed_parse_xml_file.py # preprocess the tracking-produced.xml file into .csv files for players and ball
├── pass_extraction_evaluation
│   ├── merge_and_filter_passes.py
│   ├── pass_extraction_model_eval.ipynb #Notebook with evaluation code used in the reporting
│   └── utils
│       ├── __init__.py
│       ├── build_passes.py
│       ├── evaluation.py
│       ├── filtering.py
│       ├── helpers.py
│       └── visualization.py
```
