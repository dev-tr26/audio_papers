import os
import librosa 
import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(data_path, saved_to_data_path="data/", seed=42, test_split=0.01, sort=True):
    
    metadata_path = os.path.join(data_path, "metadata.csv")
    
    metadata = pd.read_csv(metadata_path, sep="|", header=None)
    metadata.columns = ["file_path", "raw_transcript", "normalized_transcript"]
    
    metadata = metadata[["file_path", "normalized_transcript"]]
    metadata = metadata[~metadata['normalized_transcript'].isna()].reset_index(drop=True)
    
    full_path_function = lambda x: os.path.join(data_path, "wavs", f"{x}.wav")
    metadata["file_path"] = metadata["file_path"].apply(full_path_function)
    
    verify_func = lambda x: os.path.isfile(x)
    exists = metadata["file_path"].apply(verify_func)
    assert all(list(exists)), "check something is missing"
    
    duration_func = lambda x: librosa.get_duration(path=x)
    metadata["duration"] = metadata['file_path'].apply(duration_func)
    train_df, test_df = train_test_split(metadata, test_size=test_split, random_state=seed)
    
    # sorted for longest to shortest so padding is minimized 
    if sort:
        train_df = train_df.sort_values(by=["duration"],ascending=False)
        
    train_df.to_csv(os.path.join(saved_to_data_path, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(saved_to_data_path, "test_metadata.csv"), index=False)
    
    print("Completed data cleaning ")
    

if __name__ == "__main__":
    import argparse    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_ljspeech", type=str, required=True)
    parser.add_argument("--path_to_save",type=str, required=True)
    parser.add_argument("--seed", type=int,default=42)
    
    args = parser.parse_args()
    
    random_split(args.path_to_ljspeech, args.path_to_save, args.seed)
    
    