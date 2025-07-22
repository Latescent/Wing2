import csv

from pathlib import Path
from utils.helpers import read_csv
from utils.helpers import noise_level_detection

if __name__ == "__main__":
    script_directory = Path(__file__).parent.parent
    csv = read_csv(f"{script_directory}/configs/filter_params.csv")[1:]
