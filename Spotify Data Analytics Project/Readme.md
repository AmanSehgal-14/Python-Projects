
# Spotify Track Mode Prediction Project

## Project Overview
This project aims to predict whether a Spotify track is in a **major** or **minor** mode using machine learning. The mode of a song is linked to emotional context (e.g., major for happiness, minor for sadness), which can enhance recommendation systems by suggesting tracks that match the listener's mood. The workflow includes data cleaning, exploratory analysis, feature engineering, model training, hyperparameter tuning, and deployment-ready pipeline creation.

## Dataset Description
The dataset contains **113,549 tracks** with 20 audio features. Key columns include:
- `mode`: Target variable (0 = minor, 1 = major).
- `danceability`, `energy`, `acousticness`, `loudness`, `valence`, and others.
- Metadata: `track_id`, `artists`, `popularity`, `duration_ms`, etc.

**Source**: [Spotify Tracks Dataset on Hugging Face](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).

## Business Objective
Improve Spotify's recommendation system by predicting song modes to align suggestions with the user's current emotional state.

## Key Steps
1. **Data Cleaning**: Removed duplicates, null values, and irrelevant columns.
2. **Exploratory Data Analysis (EDA)**:
   - Visualized distributions of numerical features.
   - Analyzed correlations (e.g., `acousticness` vs. `energy`).
   - Observed class imbalance: **63.8% major** vs. **36.2% minor**.
3. **Feature Engineering**:
   - Encoded categorical variables (`explicit`, `track_genre`).
   - Applied feature selection using Chi-square.
4. **Model Training**:
   - Tested **Logistic Regression**, **Random Forest**, **Naive Bayes**, and **KNN**.
   - Evaluated performance using accuracy, precision, recall, and F1-score.
5. **Hyperparameter Tuning**: Improved Random Forest accuracy via grid search.
6. **Model Deployment**: Saved best models as pickle files for reuse.

## Results
- **Best Model**: Random Forest with all original features achieved **80.3% accuracy**.
- **Hyperparameters**: `max_depth=None`, `n_estimators=200`.
- Key metrics:
  - Precision: **79.5%**
  - Recall: **93.2%**
  - F1-score: **85.6%**

## Installation
### Dependencies
- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `pickle`

Install requirements:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

## Usage
1. **Load Data**:
   ```python
   df = pd.read_csv('spotify-dataset.csv')
   ```
2. **Run Jupyter Notebook**: Execute cells in `Spotify Project.ipynb` to preprocess data, train models, and save outputs.
3. **Predict Mode**:
   ```python
   # Load saved model and predict
   def predict_value(model_file, data, genre_dict):
       with open(model_file, 'rb') as f:
           model = pickle.load(f)
       # Preprocess data and predict
       return model.predict(preprocessed_data)
   ```

## Example Prediction
```python
sample = df[50:59]  # Sample data
predicted = predict_value('rfr_model_grid.pkl', sample, genre_dict)
print("Predicted Modes:", predicted)
```
**Output**: 89% accuracy on sample data.

## Recommendations
- Use the model to recommend songs with matching modes during a user's session.
- Enhance emotional consistency in playlists (e.g., upbeat tracks for major mode, melancholic for minor).

## References
1. Chase, W. (2006). *How Music Really Works!* Roedy Black Pub.
2. Google Developers. (n.d.). [Handling Imbalanced Data](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data).
3. Pandya, M. (2023). [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).

---

**Note**: Accuracy scores range from 0 to 1. Multiply by 100 for percentages.
