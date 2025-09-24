# 📊 CreditWise: Customer Credit Segmentation & Insights

**CreditWise** is an end-to-end data science project designed to analyze and segment customers using the **German Credit Data** dataset.
The project demonstrates the application of **data science in the financial sector**, including data preprocessing, exploratory data analysis (EDA), clustering, and customer insights generation.

---

## 🚀 Technologies Used
- Python 3.10.2
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

---

## 📂 Project Structure
```bash
CreditWise/
│
├── data/ <- Raw and processed datasets
│   ├── german_credit_data.csv              <- Original raw dataset
│   ├── german_credit_data_clean.csv        <- Cleaned dataset after preprocessing
│   ├── german_credit_data_clean_scaled.csv <- Scaled dataset
│   └── * cluster_*.csv, cluster_profiles.csv, etc.
│
├── notebooks/ <- Analysis & modeling notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Clustering.ipynb
│   └── 04_Alternative_Clustering
│
├── models/ <- Trained models and related objects
│   ├── kmeans_model.pkl
│   ├── standard_scaler.pkl
│   ├── label_encoders.pkl
│   ├── label_maps.json
│   ├── segment_names.json
│   └── feature_cols.json
│
├── images/ <- Visual outputs and plots
│
├── src/ <- Inference scripts 
│   ├── inference.py   <- Loads trained artifacts and predicts customer segment for new records 
│   └── __pycache__/   <- Compiled Python cache files
│
├── app.py
├── README.md
├── requirements.txt
```
---

## 📚 Dataset
- **German Credit Data**  
- Source: [Kaggle Notebook](https://www.kaggle.com/code/heidarmirhajisadati/credit-risk-prediction-using-german-credit-data)  
- Contains 1000 customer records with demographic, financial, and loan-related attributes.
- Note: This dataset version does not include a direct target variable (Risk).
- Instead, the project focuses on unsupervised learning (clustering) to identify customer groups.

---

## ⚙️ Installation & Usage
Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/emrekaratas1/CreditWise.git

# Navigate to project directory
cd CreditWise

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboard/app.py
```

## 📊 Project Goals
- Perform exploratory data analysis (EDA) to understand the dataset
- Handle missing values and encode categorical variables
- Segment customers into clusters based on financial & demographic features
- Provide insights for potential use cases in credit risk management and marketing