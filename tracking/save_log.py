import pandas as pd
import os
from datetime import datetime

def log_training_results(log_path, model_name, params, results, random_state, notes=None):
    """
    Log training results to a CSV file, now including the random state.
    
    Parameters:
    - log_path: Path to the log file.
    - model_name: Name of the model.
    - params: Dictionary of training parameters (e.g., {'epoch': 10, 'lr': 0.001}).
    - results: Dictionary of training results (e.g., {'val_loss': 0.5, 'train_loss': 0.4, 'F1': 0.8, 'accuracy': 0.75}).
    - random_state: The random state used in the training.
    - notes: Additional notes about the training, default is None.
    """
    # Default notes to an empty string if not specified
    if notes is None:
        notes = ''
    
    # Merge parameters and results dictionaries, and add model_name, random_state, and notes
    record = {**params, **results, 'random_state': random_state, 'notes': notes, 'model_name': model_name}
    
    # Convert the record to a DataFrame
    new_record_df = pd.DataFrame([record])
    
    # Check if the file exists
    if not os.path.isfile(log_path):
        # If the file does not exist, save the new record as a CSV file
        new_record_df.to_csv(log_path, index=False)
    else:
        # If the file exists, read the CSV file
        df = pd.read_csv(log_path)
        # Use pd.concat to append the new record
        updated_df = pd.concat([df, new_record_df], ignore_index=True)
        updated_df.to_csv(log_path, index=False)


def log_feature_file(log_path, file_name, file_type, note, processed_sampler_data, PROCESSED_DATA_DIR):
    """
    Log details about a feature file and save the file.
    
    Parameters:
    - log_path: Path to the log file.
    - file_name: Name of the feature file.
    - file_type: Type of the feature file (e.g., Pickle).
    - note: Notes about the feature file.
    - processed_sampler_data: The data to be saved.
    - PROCESSED_DATA_DIR: Directory where the feature file will be saved.
    """
    generate_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    record = {
        "FileName": file_name,
        "Type": file_type,
        "Note": note,
        "GenerateDate": generate_date,
    }

    new_record_df = pd.DataFrame([record])

    if not os.path.isfile(log_path):
        new_record_df.to_csv(log_path, index=False)
    else:
        df = pd.read_csv(log_path)
        updated_df = pd.concat([df, new_record_df], ignore_index=True)
        updated_df.to_csv(log_path, index=False)

    output_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    with open(output_path, 'wb') as handle:
        import pickle  # Import here if pickle is only used in this function
        pickle.dump(processed_sampler_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

