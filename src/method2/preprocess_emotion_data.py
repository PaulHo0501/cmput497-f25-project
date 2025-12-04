"""
Data Preprocessor for EmotionDynamics
This script converts your custom emotion diary CSV into the format expected by EmotionDynamics.
"""

import pandas as pd
import os
from datetime import datetime
import json

class EmotionDataPreprocessor:
    """
    Preprocesses emotion diary data for use with the EmotionDynamics framework.
    """
    
    def __init__(self, input_csv_path, output_dir="processed_data"):
        """
        Initialize the preprocessor.
        
        Args:
            input_csv_path: Path to your emotion diary CSV file
            output_dir: Directory where processed files will be saved
        """
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and validate the input CSV."""
        print(f"Loading data from {self.input_csv_path}...")
        self.df = pd.read_csv(self.input_csv_path)
        
        # Validate required columns
        required_cols = ['user_id', 'text_id', 'text', 'timestamp']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by user and timestamp
        self.df = self.df.sort_values(['user_id', 'timestamp'])
        
        print(f"Loaded {len(self.df)} entries from {self.df['user_id'].nunique()} users")
        return self
    
    def get_user_statistics(self):
        """Get statistics about users in the dataset."""
        stats = self.df.groupby('user_id').agg({
            'text_id': 'count',
            'timestamp': ['min', 'max']
        }).round(2)
        
        stats.columns = ['num_entries', 'first_entry', 'last_entry']
        stats['duration_days'] = (stats['last_entry'] - stats['first_entry']).dt.days
        
        return stats
    
    def prepare_for_ued(self, use_existing_scores=False):
        """
        Prepare data for UED analysis.
        
        Args:
            use_existing_scores: If True and valence/arousal columns exist,
                               use those instead of recalculating from text
        
        Returns:
            Dictionary with prepared data for each user
        """
        print("\nPreparing data for UED analysis...")
        
        user_data = {}
        
        for user_id in self.df['user_id'].unique():
            user_df = self.df[self.df['user_id'] == user_id].copy()
            
            # Prepare user-specific data
            user_info = {
                'user_id': int(user_id),
                'num_entries': len(user_df),
                'entries': []
            }
            
            for idx, row in user_df.iterrows():
                entry = {
                    'text_id': int(row['text_id']),
                    'text': row['text'],
                    'timestamp': row['timestamp'].isoformat(),
                    'order': len(user_info['entries'])  # Sequential order
                }
                
                # Include existing emotion scores if available and requested
                if use_existing_scores:
                    if 'valence' in row and pd.notna(row['valence']):
                        entry['valence'] = float(row['valence'])
                    if 'arousal' in row and pd.notna(row['arousal']):
                        entry['arousal'] = float(row['arousal'])
                    if 'dominance' in row and pd.notna(row['dominance']):
                        entry['dominance'] = float(row['dominance'])
                
                # Include any other available columns
                for col in ['collection_phase', 'is_words']:
                    if col in row and pd.notna(row[col]):
                        entry[col] = row[col]
                
                user_info['entries'].append(entry)
            
            user_data[user_id] = user_info
        
        return user_data
    
    def save_user_files(self, user_data, format='json'):
        """
        Save individual user data files.
        
        Args:
            user_data: Dictionary of user data from prepare_for_ued()
            format: 'json' or 'csv'
        """
        user_dir = os.path.join(self.output_dir, 'users')
        os.makedirs(user_dir, exist_ok=True)
        
        print(f"\nSaving user files to {user_dir}...")
        
        for user_id, data in user_data.items():
            if format == 'json':
                filename = os.path.join(user_dir, f'user_{user_id}.json')
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == 'csv':
                filename = os.path.join(user_dir, f'user_{user_id}.csv')
                entries_df = pd.DataFrame(data['entries'])
                entries_df.to_csv(filename, index=False)
            
        print(f"Saved {len(user_data)} user files in {format} format")
    
    def create_combined_file(self, user_data):
        """
        Create a single combined file with all users for easier processing.
        """
        filename = os.path.join(self.output_dir, 'all_users_combined.csv')
        
        all_entries = []
        for user_id, data in user_data.items():
            for entry in data['entries']:
                entry_with_user = entry.copy()
                entry_with_user['user_id'] = user_id
                all_entries.append(entry_with_user)
        
        combined_df = pd.DataFrame(all_entries)
        combined_df.to_csv(filename, index=False)
        print(f"\nSaved combined file: {filename}")
        
        return combined_df
    
    def create_text_corpus(self, user_data):
        """
        Create text corpus files for emotion lexicon analysis.
        Each user gets a text file with their entries.
        """
        corpus_dir = os.path.join(self.output_dir, 'text_corpus')
        os.makedirs(corpus_dir, exist_ok=True)
        
        print(f"\nCreating text corpus in {corpus_dir}...")
        
        for user_id, data in user_data.items():
            filename = os.path.join(corpus_dir, f'user_{user_id}_corpus.txt')
            
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in data['entries']:
                    # Write each entry on a new line with metadata
                    f.write(f"[ENTRY_{entry['text_id']}] {entry['text']}\n")
        
        print(f"Created text corpus for {len(user_data)} users")


def main():
    """
    Example usage of the preprocessor.
    """
    # Replace with your actual CSV file path
    input_csv = "data/train_subtask1.csv"  # CHANGE THIS
    
    # Initialize preprocessor
    preprocessor = EmotionDataPreprocessor(
        input_csv_path=input_csv,
        output_dir="processed_emotion_data"
    )
    
    # Load and process data
    preprocessor.load_data()
    
    # Print statistics
    print("\n=== User Statistics ===")
    stats = preprocessor.get_user_statistics()
    print(stats)
    
    # Prepare data for UED
    # Set use_existing_scores=True if you want to use your valence/arousal values
    # Set use_existing_scores=False if you want to recalculate from text using lexicons
    user_data = preprocessor.prepare_for_ued(use_existing_scores=True)
    
    # Save in multiple formats
    preprocessor.save_user_files(user_data, format='json')
    preprocessor.save_user_files(user_data, format='csv')
    
    # Create combined file
    combined_df = preprocessor.create_combined_file(user_data)
    
    # Create text corpus for lexicon-based analysis
    preprocessor.create_text_corpus(user_data)
    
    print("\n✓ Preprocessing complete!")
    print(f"  Output directory: {preprocessor.output_dir}")
    print(f"  Total users: {len(user_data)}")
    print(f"  Total entries: {len(combined_df)}")


if __name__ == "__main__":
    main()
