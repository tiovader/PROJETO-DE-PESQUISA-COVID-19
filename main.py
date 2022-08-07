from feature_extractor import FeatureExtractor
import os
import pandas as pd

def main():
    metadata = pd.read_csv('./images/metadata.csv')
    metadata['path'] = metadata.apply(lambda x: os.path.join('./images', x.folder, x.filename), axis=1)
    features = FeatureExtractor.from_dataframe(metadata)
    df = pd.merge(metadata, features, left_index=True, right_index=True)   
    to_drop = ["filename", "dataset", "path"]
    columns = df.columns.difference(to_drop).to_list()
    df["covid"] = df["covid"].astype("uint8")
    df.drop_duplicates(subset=columns, inplace=True)
    df.to_csv("./features.csv", index=False)
    


if __name__ == "__main__":
    main()
