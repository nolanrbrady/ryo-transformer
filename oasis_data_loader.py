import nibabel as nib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import T5Tokenizer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms
import torchio as tio
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class OASISDataLoader:
    def __init__(self, batch_size=32, max_text_length=50, train_size=0.6, test_size=0.4, val_size=0.5):
        self.csv_path = "./OASIS-2/OASIS-2_Highly_Detailed_MRI_Visit_Descriptions.csv"
        # self.csv_path = "./OASIS-2/OASIS-2_Varied_Verbose_Descriptions.csv"
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.train_size = train_size
        self.test_size = test_size  
        self.val_size = val_size
        self._init_transforms()

    def _init_transforms(self):
        self.train_transform = tio.Compose([
            tio.ToCanonical(),  # Ensure consistent orientation
            tio.RescaleIntensity(out_min_max=(0, 1)),  # Normalize intensity
            tio.RandomFlip(axes=('LR',)),  # Randomly flip left-right
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Random affine transformations
        ])
        
        self.test_transform = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ])

    def _preprocess_data(self, df):
        df = df.copy()
        df = df.dropna(subset=['MMSE'])
        print("Columns in the dataframe: ", df.columns)
        
        # Normalize continuous features
        cont_features = ['Age', 'EDUC', 'eTIV', 'nWBV', 'ASF']
        df[cont_features] = StandardScaler().fit_transform(df[cont_features])
        
        # Encode categorical variables
        df['M/F'] = df['M/F'].map({'M': 0, 'F': 1})
        df['Hand'] = df['Hand'].map({'R': 0, 'L': 1, 'A': 2})
        df['CDR'] = df['CDR'].astype(float)
        df['SES'] = df['SES'].fillna(df['SES'].median())
        
        # Map groups and exclude Converted scans
        df['Group'] = df['Group'].map({'Nondemented': 0, 'Demented': 1, 'Converted': 2})
        df = df[df['Group'] != 2]  # This line filters out Converted scans
        
        # Create MMSE bins
        df['MMSE_bin'] = pd.cut(df['MMSE'], 
                              bins=[0, 10, 20, 26, 30], 
                              labels=['severe', 'moderate', 'mild', 'normal'])
        
        # Precompute the categorical codes and store in a new column
        df['MMSE_bin_codes'] = df['MMSE_bin'].cat.codes
        
        # Correct for skew in the MMSE data
        power_transformer = PowerTransformer(method='yeo-johnson')
        mmse_transformed = power_transformer.fit_transform(df[['MMSE']])

        # Step 2: Standard Scaling (Z-score normalization)
        scaler = StandardScaler()
        df['MMSE_transformed'] = scaler.fit_transform(mmse_transformed)
        
        return df

    def _create_datasets(self, df):
        """
        Splits the dataframe into training, testing, and validation datasets by ensuring that all scans
        from each subject appear exclusively in one split. This prevents data leakage where, for example,
        a subject's initial scan appears in training and their follow-up scan appears in validation.
        """
        # Get all unique subjects
        unique_subjects = df['Subject ID'].unique()
        
        # First split the subjects into a training set and a temporary set (which will be split into test and val)
        train_subjects, temp_subjects = train_test_split(unique_subjects, test_size=0.4, random_state=42)
        
        # Split the temporary subjects equally into test and validation sets
        test_subjects, val_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)
        
        # Create boolean masks for each split based on Subject IDs
        train_mask = df['Subject ID'].isin(train_subjects)
        test_mask = df['Subject ID'].isin(test_subjects)
        val_mask = df['Subject ID'].isin(val_subjects)
        
        return (
            OASISDataset(df[train_mask], self.tokenizer, self.train_transform, self.max_text_length),
            OASISDataset(df[test_mask], self.tokenizer, self.test_transform, self.max_text_length),
            OASISDataset(df[val_mask], self.tokenizer, self.test_transform, self.max_text_length)
        )

    def get_dataloaders(self):
        df = pd.read_csv(self.csv_path)
        df = self._preprocess_data(df)
        print("Data in the dataloader:", df.columns)
        train_ds, test_ds, val_ds = self._create_datasets(df)
        
        # Use the original DataFrame with the appropriate mask to get subject IDs
        train_subject_ids = df.loc[train_ds.mask, 'Subject ID']
        test_subject_ids = df.loc[test_ds.mask, 'Subject ID']
        val_subject_ids = df.loc[val_ds.mask, 'Subject ID']
    
        
        return (
            DataLoader(train_ds, batch_size=self.batch_size, sampler=GroupSampler(train_subject_ids), collate_fn=self.collate_fn),
            DataLoader(test_ds, batch_size=self.batch_size, sampler=GroupSampler(test_subject_ids), collate_fn=self.collate_fn),
            DataLoader(val_ds, batch_size=self.batch_size, sampler=GroupSampler(val_subject_ids), collate_fn=self.collate_fn)
        )

    def collate_fn(self, batch):
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'mmse': torch.stack([item['mmse'] for item in batch]),
            'mmse_bin': torch.stack([item['mmse_bin'] for item in batch]),
            'mmse_transformed': torch.stack([item['mmse_transformed'] for item in batch]),
            'decoder_input_ids': torch.stack([item['decoder_input_ids'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'group': torch.stack([item['group'] for item in batch])
        }

class OASISDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=None, max_text_length=50):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_length = max_text_length
        self.mask = dataframe.index  # Store original indices for GroupSampler

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load MRI image from nibabel.
        # The image is expected to have shape (256, 256, 128, 1):
        #   - 256: Height
        #   - 256: Width
        #   - 128: Depth (number of slices)
        #   - 1: Channel
        img = nib.load(row['MRI-URL']).get_fdata(caching='unchanged')  # Add memory mapping
        img = torch.as_tensor(np.asarray(img, dtype=np.float32))  # Use memory-efficient conversion
        img = img.permute(3, 2, 0, 1)  # Now shape: (1, 128, 256, 256)
        
        if self.transform:
            img = self.transform(img)
            
        # Use the text description to generate caption tokens.
        text = str(row['Generated_descriptions'])
        inputs = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Shift decoder inputs for teacher forcing
        decoder_input_ids = inputs.input_ids.clone()
        decoder_input_ids[decoder_input_ids == self.tokenizer.pad_token_id] = -100
        labels = decoder_input_ids.clone()
        decoder_input_ids = decoder_input_ids[:, :-1]
        
        # When processing labels:
        labels = labels.squeeze()
        if len(labels) > self.max_text_length:
            labels = labels[:self.max_text_length]  # Truncate
        else:
            labels = F.pad(labels, (0, self.max_text_length - len(labels)), value=self.tokenizer.pad_token_id)

        return {
            'image': img,
            'vocab_size': self.tokenizer.vocab_size,
            'mmse': torch.tensor(row['MMSE'], dtype=torch.float32),
            'mmse_transformed': torch.tensor(row['MMSE_transformed'], dtype=torch.float32),
            'mmse_bin': torch.tensor(row['MMSE_bin_codes'], dtype=torch.long),
            'decoder_input_ids': decoder_input_ids.squeeze(0),
            'labels': labels,
            'group': torch.tensor(row['Group'], dtype=torch.long)
        }

class GroupSampler(Sampler):
    def __init__(self, groups):
        self.groups = groups
        self.unique_groups = groups.unique()
        
    def __iter__(self):
        return iter(torch.randperm(len(self.unique_groups)).tolist())
        
    def __len__(self):
        return len(self.unique_groups)
    

# Test the file
# Initialize data loader
data_loader = OASISDataLoader(
    batch_size=16,
    max_text_length=300
)

# Get dataloaders
train_loader, test_loader, val_loader = data_loader.get_dataloaders()

# Print the first batch of the train loader
for batch in train_loader:
    print(batch['mmse'].shape)
    print(batch['image'].shape)
    break