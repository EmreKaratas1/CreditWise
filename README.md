# 📊 CreditWise: Customer Credit Segmentation & Insights

**CreditWise** is an end-to-end data science project designed to analyze and segment customers using the **German Credit Data** dataset.
The project demonstrates the application of **data science in the financial sector**, including data preprocessing, exploratory data analysis (EDA), clustering, and customer insights generation.

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
├── data/ <- Raw dataset
│   └── german_credit_data.csv
│
├── notebooks/ <- Analysis & modeling notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Clustering.ipynb
│
├── models/ <- Trained models (to be added later)
│
├── dashboard/ <- Streamlit application (planned)
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

## 📈 Results (to be updated)
- Visual insights from EDA (distributions, correlations)
- Cleaned dataset ready for modeling
- Customer clusters with interpretation (e.g., low-risk / high-risk groups)
- Documentation of key insights

## 🔮 Future Work
- Experiment with different clustering algorithms (K-Means, Hierarchical, DBSCAN)
- Compare with supervised approaches using alternative datasets (with Risk variable)
- Develop a Streamlit dashboard for interactive customer segmentation
- Explore integration with BI tools (Power BI, Tableau) for reporting