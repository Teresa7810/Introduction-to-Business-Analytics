# EV Charging Sessions: Business Analytics Project

We analyse public EV charging sessions in Palo Alto from 2011 to 2020 for the Introduction to Business Analytics course. The work covers data cleaning, exploratory analysis, driver clustering, demand forecasting, and a study of hoarding, where a car stays plugged in after it has finished charging.

## Run order

The raw data is Data/EVChargingStationUsage.csv. Run project.ipynb first, since every other notebook reads the cleaned files it produces.

1. **project.ipynb**: data cleaning and pre-processing. Reads Data/EVChargingStationUsage.csv and writes Data/ev_clean.pkl and Data/station_active.pkl.
2. **EDA_executed.ipynb**: exploratory data analysis. Reads Data/ev_clean.pkl.
3. **2. Data Modelling.ipynb**: driver clustering with K-Means, DBSCAN and Ward. Reads Data/ev_clean.pkl and uses the helper functions in Functions.py and Visualizations.py.
4. **3. Prediction.ipynb**: demand forecasting. Reads Data/ev_clean.pkl and Data/station_active.pkl.
5. **4. Exploratory.ipynb**: hoarding, from prediction at plug-in to business impact. Reads Data/ev_clean.pkl and writes Data/grid_results.pkl.

Steps 2 to 5 can run in any order. The .pkl files are in .gitignore, so run project.ipynb again after cloning the repository or pulling a change to it.

## Notebooks and paper sections

- Section 2 (Data): project.ipynb and EDA_executed.ipynb
- Section 3 (Prediction challenge): 2. Data Modelling.ipynb and 3. Prediction.ipynb
- Section 4 (Exploratory component): 4. Exploratory.ipynb

## Requirements

Python 3.11 or later, with these packages:

```
pandas numpy matplotlib seaborn scikit-learn scipy plotly umap-learn
```

## Notes

User ID 0 is a shared default account. Its 447 sessions (2011 to 2015) are spread over 14 stations, and some of them overlap at different stations, so they cannot belong to one driver. project.ipynb keeps these sessions but treats the ID as missing.

In 4. Exploratory.ipynb, setting RUN_GRID = True runs the hyperparameter grid again, which takes 3 to 7 minutes, and saves the results to Data/grid_results.pkl. With RUN_GRID = False the notebook loads that file instead. Every model uses random_state=42, so a new run gives the same results.
