# Music Genre Classification using Statistical and Machine Learning

This project presents a comparative study of various machine learning and deep learning architectures for automatic music genre classification. By analyzing acoustic features and visual representations of audio, the system aims to automate content organization for radio stations, producers, and music streaming services.

## Project Overview

The core challenge involves accurately assigning genre labels (e.g., rock, jazz, or classical) to audio signals. This is complex because genres are often subjective and share overlapping acoustic features such as tempo, rhythm, and instrumentation.

### Core Objectives
* Streamlining radio stations and genre-based playlists.
* Providing music analysis and industry insights.
* Aiding producers and musicians in audio production.
* Assisting with music licensing and copyrights.

---

## Technical Methodology

### Dataset and Pre-processing
The models are trained and evaluated using the **GTZAN genre collection**, which contains 1,000 audio samples across 10 distinct genres.
* **Feature Extraction**: Extracted tempo, spectral centroid, and zero-crossing rate into 3-second and 30-second segments.
* **Acoustic Features**: Utilized Mel-frequency Cepstral Coefficients (MFCCs) to capture spectral and temporal characteristics perceptually uniform to human hearing.
* **Visual Representation**: Transformed raw audio into **mel-spectrograms**, enabling the application of computer vision techniques for classification.



### Evaluated Models
1. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies samples based on a majority vote of their neighbors.
2. **Logistic Regression**: A statistical technique adapted for multinomial classification across the 10 genre classes.
3. **Decision Trees**: A model that recursively partitions the feature space into interpretable regions based on input values.
4. **Convolutional Neural Networks (CNN)**: Deep learning models designed to learn hierarchical feature representations directly from spectrogram images.
5. **Recurrent Convolutional Neural Network (RCNN)**: A hybrid architecture combining CNNs for spatial feature extraction with GRU layers to learn sequential and temporal patterns.

---

## Results and Performance

The performance of each algorithm was evaluated based on test accuracy and detailed classification metrics.

| Algorithm Choice | Test Accuracy | Performance Note |
| :--- | :--- | :--- |
| **Decision Tree** | 53% | Lower accuracy due to potential overfitting and limited generalization. |
| **Logistic Regression** | 66% | Baseline performance for linear probabilistic classification. |
| **KNN** | 67% | Effective for smaller datasets but computationally expensive. |
| **CNN** | 86% - 87.5% | High efficacy in capturing hierarchical patterns from mel-spectrograms. |
| **RCNN** | **89%** | Best performing model due to modeling both spatial and sequential dependencies. |



---

## Conclusion

The project demonstrates that deep learning approaches—specifically **CNNs** and **RCNNs**—significantly outperform traditional statistical methods in music genre classification. While traditional models like KNN and Logistic Regression provide strong baselines, the ability of neural networks to process visual mel-spectrograms allows for superior accuracy in distinguishing between acoustically similar and subjective genres.

---
**Authors**: Mehul Agarwal & Rahul Omalur Ramesh (Indraprastha Institute of Information Technology Delhi).
