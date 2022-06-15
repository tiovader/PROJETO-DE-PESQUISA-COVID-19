import features as ft
import pandas as pd
import os


def get_path(row: pd.DataFrame):
    return os.path.join(".", "images", row.folder, row.filename)


def main():
    metadata = pd.read_csv("./csv/metadata.csv")
    metadata["path"] = metadata.apply(get_path, axis=1)
    metadata.drop(columns=["folder"], inplace=True)

    features = ft.from_dataframe(metadata)
    df = pd.merge(metadata, features, left_index=True, right_index=True)

    to_drop = ["filename", "dataset", "path"]
    columns = df.columns.difference(to_drop).to_list()
    df.drop_duplicates(subset=columns, inplace=True)
    df.to_csv("./csv/features.csv", index=False)


if __name__ == "__main__":
    main()
