# Dataset

The full FloodingDataset2 is hosted on Google Drive due to size constraints.

## Download Dataset

1. Download from: https://drive.google.com/drive/folders/1PXc9VTeQgV5WeNxa2NOC9kbRpnQtg20o?usp=sharing
2. Extract to: `/content/drive/MyDrive/FloodingDataset2/`

## Dataset Structure
```
FloodingDataset2/
├── StreetFloodClasses/
│   ├── MajorFlood/
│   ├── MinorFlood/
│   ├── ModerateFlood/
│   ├── NoFlood/
│   └── parks_walkways/
├── junk/
│   ├── Cats.zip
│   ├── Dogs.zip
│   ├── Swimmingpool.zip
│   └── ...
└── processed_data/
    └── binary/
        ├── train/
        ├── val/
        └── test/
```

## Dataset Statistics

See `metadata/` folder for:
- Class distribution
- Train/val/test splits
- Dataset statistics

Total images: 3,754
- Train: 2,627 (70%)
- Val: 563 (15%)
- Test: 564 (15%)

# Dataset (too large for GitHub)
data/FloodingDataset2/
data/processed_data/binary/
data/processed_data/multiclass/
data/extracted/junk/

# Models (use releases instead)
models/*.keras
