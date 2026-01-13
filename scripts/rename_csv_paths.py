from backflip.utils import rename_csv_paths
from argparse import ArgumentParser
import pandas as pd

if __name__ == "__main__":

    parser = ArgumentParser(description="Rename metadata CSV paths to absolute paths and overwrite the CSV file.")
    parser.add_argument("paths_to_csv", type=str, nargs="+", help="Path(s) to the CSV file(s) to be processed.")
    args = parser.parse_args()

    for path_to_csv in args.paths_to_csv:
        renamed_df = rename_csv_paths(path_to_csv)
        renamed_df.to_csv(path_to_csv, index=False)
