# House Price Prediction

### Overview

This is a project that aims to build models to predict the residential house price from house features

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

I'll use this public dataset to highlight some of key tasks involed in a data driven project:

exploratory data analysis
data wrangling
feature engineering
building ML models (regularized regression, decision tree, bagging and boosting models)
feature selection and interpretation
hyperparameters tuning
model selection

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [Jupyter Notebook](http://ipython.org/notebook.html)

### Notebook

Code for data handling & transformation, building ML models and creating visulizations are all included in the notebook. 
[house_price_JK.ipynb](https://github.com/JK558/house-price-prediction/blob/master/house_price_JK.ipynb)
It also includes discussion of the project in greater detail. Please feel free to explore this file.

### Data

The dataset include 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. 

- [train.csv](https://github.com/JK558/house-price-prediction/blob/master/train.csv) - the training set
- [test.csv](https://github.com/JK558/house-price-prediction/blob/master/test.csv) - the test set
- [data_description.txt](https://github.com/JK558/house-price-prediction/blob/master/data_description.txt) - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here

Here's a brief version of what you'll find in the data description file.

**Target Variable**

- `SalePrice` - the property's sale price in dollars. 

**Features**

**Features**

- `MSSubClass`: The building class
- `MSZoning`: The general zoning classification
- `LotFrontage`: Linear feet of street connected to property
- `LotArea`: Lot size in square feet
- `Street`: Type of road access
- `Alley`: Type of alley access
- `LotShape`: General shape of property
- `LandContour`: Flatness of the property
- `Utilities`: Type of utilities available
- `LotConfig`: Lot configuration
- `LandSlope`: Slope of property
- `Neighborhood`: Physical locations within Ames city limits
- `Condition1`: Proximity to main road or railroad
- `Condition2`: Proximity to main road or railroad (if a second is present)
- ...


see [data_description.txt](https://github.com/JK558/house-price-prediction/blob/master/data_description.txt)  for more
