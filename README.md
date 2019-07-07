## Motivation 
In this project. I used a big dataset I scraped before of a few thousand perfume links for their characteristics, like perfume name, designer, fragrance family, main accords, average vote, people's opinions and willingness to purchase, in numbers. etc. 
My goal is two fold, 
a. to create a predictive model for the perfumes that didn't have an average vote, 
b. to give insight and business recommendations to designers on perfumes that sell and are well received globally. 

**Data** <br />
I have a previous repository named "recommender_system". In there, I used the reviews for 500 of the perfumes instead of the whole ~5000 of them for quicker handling and showcase. Here, I use the characteristics of the ~5000 perfumes.

## Technical details
I stard with cleaning and preparing the data frame, in the notbook called `perfume_chars_preparation.ipynb`
- The primitive collection of data is in a separate folder named `collecting_data`. 
- The analysis in the notebook called `perfume_chars_analysis.ipynb` where I did essential Exploratory Data Analysis, and ran regression analysis to predict ratings for a set of perfumes, using Lasso regularization, and feature engineering. The best model I got was Lasso, with Polynomial features of the second degree, and selecting best 200 features. With 80% accuracy. 
- in `perfume_chars_clustering.ipynb` I attempted to cluster the dataset, looking for groups of data points, which would shed more light on perfumes tend to group. The data didn't appear to be in any way grouped, and I don't think there's a cluster. 
- In `perf_chars_anal_classification.ipynb`, I changed the regression problem (predicting a continuous value for ratings between 1 and 5) to a classification problem, where I made ratings in bins of width 0.5 This approach yielded much better accuracy of 0.937 
