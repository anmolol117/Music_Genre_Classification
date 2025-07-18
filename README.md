# Music_Genre_Classification
This project presents an end-to-end system for automatic music genre classification based solely on audio content. We trained multiple machine learning models, including a robust deep neural network, on a dataset of 30-second music clips with 57 pre-extracted audio features (such as spectral centroid, chroma, and zero-crossing rate). These models were used to predict the genre of music across ten categories.

The system includes preprocessing, model training, evaluation using confusion matrices and accuracy metrics, and model serialization for deployment.

A user-friendly Python interface allows users to select an audio file directly from the file explorer. Upon selection, the system extracts all relevant features from the audio file, scales them, feeds them into the trained neural network, and predicts the genre in real-time. Additionally, the selected track is played for 10 seconds to enhance user interactivity.

The application combines audio signal processing, deep learning, and GUI design to showcase the potential of AI in music analysis and classification.
