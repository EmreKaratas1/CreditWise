# 📊 CreditWise: Customer Credit Risk Prediction & Segmentation

**CreditWise** is an end-to-end data science project designed to predict customer credit risk and perform customer segmentation using the **German Credit Data** dataset.  
The project showcases the application of **data science in the financial sector**, including data preprocessing, exploratory data analysis (EDA), predictive modeling, and dashboard development.

---

## 🚀 Technologies Used
- Python 3.10.2
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit (for interactive dashboard)
- Jupyter Notebook

---

## 📂 Project Structure
```bash
CreditWise/
│
├── data/ <- Raw and cleaned datasets
│   ├── german_credit_data.csv
│   └── german_credit_data_clean.csv
│
├── notebooks/ <- Analysis & modeling notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Modeling.ipynb
│   └── 03_Clustering.ipynb
│
├── models/ <- Trained models (.pkl, .joblib)
│
├── dashboard/ <- Streamlit application
│   └── app.py
│
├── images/ <- Visual outputs and plots
│
├── README.md
└── requirements.txt
```
---

## 📚 Dataset
- **German Credit Data**  
- Source: [Kaggle Notebook](https://www.kaggle.com/code/heidarmirhajisadati/credit-risk-prediction-using-german-credit-data)  
- Includes both raw (`german_credit_data.csv`) and cleaned (`german_credit_data_clean.csv`) versions for reproducibility.

---

## ⚙️ Installation & Usage
Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/emrekaratas516/CreditWise.git

# Navigate to project directory
cd CreditWise

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboard/app.py
```

## 📊 Project Goals
- Predict customer credit risk using supervised machine learning models
- Segment customers into groups based on financial & demographic features
- Demonstrate risk management and customer analytics applications in banking

## 📈 Results (to be updated)
- Exploratory data analysis (EDA) with visual insights
- Model training & evaluation (e.g., accuracy, ROC curve, confusion matrix)
- Customer clustering for segmentation analysis
- Interactive Streamlit dashboard for real-time exploration

## 🔮 Future Work
- Improve performance by testing advanced ML algorithms (XGBoost, LightGBM)
- Test with larger and more diverse datasets
- Containerization with Docker for deployment
- Integration with Power BI or Tableau for business reporting