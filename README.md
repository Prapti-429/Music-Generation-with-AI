# Music Generation with AI

This project is an AI-powered music generation system that composes original music using deep learning techniques, specifically Recurrent Neural Networks (RNNs). The model is trained on MIDI files and can generate new music sequences.

## Folder Structure
- **data/**: Contains music datasets in MIDI format.
- **models/**: Stores trained models.
- **src/**: Contains the source code for data loading, model architecture, and training.
- **requirements.txt**: List of dependencies.
- **.gitignore**: Specifies files and folders to ignore in Git.

Music-Generation-AI/
├── data/                 # Folder for storing music data (e.g., MIDI files)
├── models/               # Folder for saving trained models
├── src/                  # Source code for the project
│   ├── data_loader.py    # Data loading and preprocessing script
│   ├── model.py          # AI model architecture (RNN)
│   └── train.py          # Script to train the model and generate music
├── requirements.txt      # List of dependencies
├── README.md             # Project overview
└── .gitignore            # Files to ignore in Git
