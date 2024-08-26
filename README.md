# Genre Classification Model

This project implements a deep learning model to classify music genres based on various audio features.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`
5. Place your dataset in the `data/` directory
6. Run the model: `python main.py`

## Project Structure

- `data/`: Directory for storing the dataset
- `models/`: Directory for saving trained models
- `src/`: Source code for the project
- `tests/`: Unit tests (to be implemented)
- `main.py`: Main script to run the project
- `requirements.txt`: List of Python dependencies

## Project Timeline:

- Data gathering: Used finally spotify_data.csv.  
- EDA: Normalized numerical features, eliminated various genres, transformed categorical features, erased irrelevant data.
- Model: not more then 30% of accuracy reported so we are going to try with less genres.



21/8: tried t-sne and hierarchical classification with 5 genres 78%
10best genres 50%
first unified genres 100%
second unified (better unified) 50%


Dataset:
Class distribution:
Class acoustic: 1158.0
Class afrobeat: 704.0
Class alt-rock: 1140.0
Class ambient: 1230.0
Class black-metal: 1168.0
Class blues: 1087.0
Class breakbeat: 635.0
Class cantopop: 873.0
Class chicago-house: 284.0
Class chill: 1052.0
Class classical: 986.0
Class club: 810.0
Class comedy: 1055.0
Class country: 1009.0
Class dance: 964.0
Class dancehall: 1051.0
Class death-metal: 995.0
Class deep-house: 942.0
Class detroit-techno: 205.0
Class disco: 935.0
Class drum-and-bass: 795.0
Class dub: 1016.0
Class dubstep: 286.0
Class edm: 636.0
Class electro: 665.0
Class electronic: 516.0
Class emo: 1149.0
Class folk: 915.0
Class forro: 1066.0
Class french: 1032.0
Class funk: 851.0
Class garage: 957.0
Class german: 929.0
Class gospel: 1245.0
Class goth: 984.0
Class grindcore: 792.0
Class groove: 744.0
Class guitar: 943.0
Class hard-rock: 771.0
Class hardcore: 841.0
Class hardstyle: 697.0
Class heavy-metal: 824.0
Class hip-hop: 897.0
Class house: 282.0
Class indian: 1146.0
Class indie-pop: 558.0
Class industrial: 814.0
Class jazz: 917.0
Class k-pop: 1121.0
Class metal: 371.0
Class metalcore: 358.0
Class minimal-techno: 797.0
Class new-age: 1074.0
Class opera: 804.0
Class party: 572.0
Class piano: 710.0
Class pop: 360.0
Class pop-film: 941.0
Class power-pop: 940.0
Class progressive-house: 559.0
Class psych-rock: 753.0
Class punk: 329.0
Class punk-rock: 387.0
Class rock: 179.0
Class rock-n-roll: 898.0
Class romance: 344.0
Class sad: 312.0
Class salsa: 970.0
Class samba: 999.0
Class sertanejo: 976.0
Class show-tunes: 609.0
Class singer-songwriter: 769.0
Class ska: 790.0
Class sleep: 963.0
Class songwriter: 29.0
Class soul: 467.0
Class spanish: 1037.0
Class swedish: 653.0
Class tango: 829.0
Class techno: 464.0
Class trance: 469.0
Class trip-hop: 616.0

# Define genre mapping
Read in mapping.py

23/08:
New class distribution:
Class Electronic/Dance: 223029 (19.55%)
Class Rock/Metal: 200013 (17.53%)
Class World Music: 135633 (11.89%)
Class Other: 125521 (11.00%)
Class Country/Folk: 85076 (7.46%)
Class Pop/Mainstream: 69195 (6.07%)
Class Traditional: 68963 (6.05%)
Class Reggae/Ska: 51349 (4.50%)
Class Latin: 51169 (4.49%)
Class Ambient/Chill: 40295 (3.53%)
Class Hip-Hop/R&B: 35962 (3.15%)
Class Instrumental: 31050 (2.72%)
Class Miscellaneous: 23425 (2.05%)

Final balanced class distribution:
Class Country/Folk: 223029 (7.69%)
Class World Music: 223029 (7.69%)
Class Rock/Metal: 223029 (7.69%)
Class Ambient/Chill: 223029 (7.69%)
Class Traditional: 223029 (7.69%)
Class Electronic/Dance: 223029 (7.69%)
Class Pop/Mainstream: 223029 (7.69%)
Class Other: 223029 (7.69%)
Class Reggae/Ska: 223029 (7.69%)
Class Instrumental: 223029 (7.69%)
Class Miscellaneous: 223029 (7.69%)
Class Latin: 223029 (7.69%)


tried pipeline instead of just smote: remove kfold from model and try new balancing

just obtained 6.25%