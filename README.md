# Virtual Data Science with Python Internship - Saloni Kumari

Python code for the six weekly tasks. Each script runs on its own and saves its figures to a `figures/` folder.

| Week | Topic | Dataset | Script |
|---|---|---|---|
| 1 | Data acquisition, cleaning, preprocessing | Breast Cancer Wisconsin (UCI, via scikit-learn) | Week1_Data_Cleaning_Report.py |
| 2 | Exploratory data analysis and visualization | Diabetes progression (442 patients) | Week2_EDA_Report.py |
| 3 | Unsupervised learning and clustering | Wine (UCI, 178 wines) | Week3_Clustering_Report.py |
| 4 | Supervised learning (classification) | Breast Cancer Wisconsin | Week4_Supervised_Learning_Report.py |
| 5 | Deep learning (NumPy neural network) | Digits (UCI, 1,797 images) | Week5_Deep_Learning_Report.py |
| 6 | Capstone: full pipeline | Diabetes progression | Week6_Capstone_Report.py |

## Run
```
pip install numpy pandas matplotlib seaborn scikit-learn scipy
python Week1_Data_Cleaning_Report.py
```
Note: in Week 1, some defects (missing values, duplicates, impossible values) are injected into a copy of the data on purpose, because the original dataset has none. This is explained in the report.
