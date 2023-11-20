# ENM3440HW3
## Question
I was initially interested in answering Lizzy’s question:

A trend in the past two decades has been the hyper-acceleration of material that students are learning before they arrive at college. AP Calc BC used to be the farthest a student could accelerate their math learning in high school, but now an increasing number of students are taking linear algebra and multivariable calculus in high school. 

Question: Does the hyper-acceleration of educational curriculum actually help students learn more and be better prepared for college?

After searching around for some relevant datasets, I modified the question to better fit what was possible with the scope of this project. 

Modified Question: How does going to a magnet school affect your level of college readiness in the state of California?

### Contextualizing the Modifications
To summarize an article published by Goodwin University:
Magnet schools, as specialized public schools with theme-based curricula, offer several benefits that prepare students for college and beyond. They foster a passion for learning by allowing students to focus on areas of interest in a hands-on, experiential learning environment. Overall, magnet schools are designed to develop skills needed for college success, immersing students in an environment that mirrors the structure and demands of college-level courses​. 

Read more about magnet schools here: \
[Goodwin University Article](https://www.goodwin.edu/enews/does-a-magnet-school-look-good-for-college/#:~:text=Magnet%20school%20helps%20students%2C%20particularly,might%20mirror%20their%20college%20schedule.) \
[Documented Gaps Between School Types](https://www.montgomeryadvertiser.com/story/news/education/2018/06/26/act-scores-college-career-readiness-gap-highlight-mps-magnet-non-magnet-disparity/732382002/)

It was for the aforementioned reasons that I decided magnet schools would be an appropriate measure of acceleration in curriculum. Clearly, there are some differences between attending a magnet school and simply accelerating your curriculum or taking more advanced classes, but given that magnet schools tend to accelerate curriculum as a given, and the lack of data for individual students choosing to accelerate curriculum, I found that this was the best way to move forward with the project.

## Introduction

The purpose of this analysis was to explore the impact of attending a magnet school on a student's likelihood of achieving higher educational attainment, while also considering the potential confounding effect of wealth by county. Magnet schools, known for their specialized curricula and diverse student bodies, are often regarded as catalysts for enhanced educational outcomes. However, discerning their true impact requires a nuanced approach that accounts for various socio-economic factors. This study utilized datasets including the Common Core of Data (CCD) for school characteristics, geocode data for geographic information, educational attainment data, and wealth data indicated by median household income.

## Dataset Overview

### NCES CCD Data for School Characteristics
The National Center for Education Statistics is a government organization that collects and publishes data on public and private educational institutions in the US. From the website: 

The primary purposes of the Public Elementary/Secondary School Universe Survey are:
to provide a complete listing of all public elementary and secondary schools in the country;
to provide basic information and descriptive statistics on all schools, their students, and their teachers
This dataset provided a list of elementary, middle, and high schools in the USA, along with information about whether they were a magnet school, and some other geographical information.
Further documentation and additional files can be found here: [NCES Characteristic Data](https://nces.ed.gov/ccd/pubschuniv.asp)

### State of California Educational Attainment Data
This data was provided by the State of California. From the website:
The educational attainment table is part of a series of indicators in the Healthy Communities Data and Indicators Project (HCI) of the Office of Health Equity. The goal of HCI is to enhance public health by providing data, a standardized set of statistical measures, and tools that a broad array of sectors can use for planning healthy communities and evaluating the impact of plans, projects, policy, and environmental changes on community health.
With  this data, I was able to assess the level of educational attainment on a county by county basis. For my analysis, I used levels of educational attainment as a proxy for the level to which certain groups were prepared for college, which is another limitation of the study. 

For additional information: [Educational Attainment Data](https://catalog.data.gov/dataset/educational-attainment-f4293)

### NCES EDGE GeoCode Data

From the website:
NCES relies on information about school location to help construct school-based surveys, support program administration, identify associations with other types of geographic entities, and to help investigate the social and spatial context of education. EDGE creates and assigns address geocodes (estimated latitude/latitude values) and other geographic indicators to public schools, public local education agencies, private schools, and postsecondary schools. The geographic data are provided as shapefiles, and basic attribute data are available as Excel and SAS tables. 

The use of this data was necessary for merging the CCD and educational attainment data, as there were no commonalities between the two that would have been easy to match up together. 

For additional information: [GeoCode Data](https://nces.ed.gov/programs/edge/Geographic/SchoolLocations) 

### NIH HDPulse Wealth Data

One of the main confounders I was hoping to tackle in this project was wealth as a predictor of educational attainment. In order to accurately do this, I needed to incorporate wealth by county data into my analysis as well. 

For additional information: [NIH HDPulse](https://hdpulse.nimhd.nih.gov/data-portal/social/table?race=00&race_options=race_7&sex=0&sex_options=sexboth_1&age=001&age_options=ageall_1&demo=00011&demo_options=income_3&socialtopic=030&socialtopic_options=social_6&statefips=06&statefips_options=area_states&function=getTable&path=%2Fsocial)


## Data Analysis
### Data Filtration, Cleaning, and Merging
Merging CCD and Geocode Data: The Common Core of Data (CCD), containing information about schools, including their magnet status, was merged with geocode data to link schools with geographical locations. The key column for merging was NCESSH, which is a unique identifier given to each school by the NCES. 

Incorporating Educational Attainment Data: This merged dataset was then combined with educational attainment data, focusing on the percentage of the population with higher education degrees in corresponding counties. The merge happened over the association of each entry with a FIPS (Federal Information Processing Standard) number  in both datasets, which is assigned on a county level. 

	Code:
 ```
# Redoing the initial merge based on NCESSCH
ccd_magnet_geocode_merged = pd.merge(ccd_magnet_schools, geocode_data, on="NCESSCH", how='inner')

# Preprocessing the Educational Attainment Data
# Averaging out rows that are duplicated in the county_name column
educational_attainment_avg = educational_attainment_data.groupby('county_name').mean().reset_index()

# Checking the columns in the merged CCD-Geocode dataset and Educational Attainment dataset
columns_ccd_geocode = ccd_magnet_geocode_merged.columns
columns_educational_attainment = educational_attainment_avg.columns

columns_ccd_geocode, columns_educational_attainment
```

Adding Wealth Data: The dataset was further enriched by merging it with wealth data, using median household income as a proxy for wealth by county. Again, the merge was done by FIPS number. 

Code:
```
# Skipping the initial rows and renaming the columns in the Wealth Data
wealth_data_cleaned = wealth_data.iloc[3:].rename(columns={
    'Income (Median household income) for California by County': 'County',
    'Unnamed: 1': 'FIPS',
    'Unnamed: 2': 'Median_Income',
    'Unnamed: 3': 'Income_Rank_US',
    'Unnamed: 4': 'Unused_Column'
})

# Dropping the first row which contains the column names
wealth_data_cleaned = wealth_data_cleaned.iloc[1:]

# Converting FIPS column to float for compatibility with county_fips in the merged dataset
wealth_data_cleaned['FIPS'] = wealth_data_cleaned['FIPS'].astype(float)

# Displaying the first few rows of the cleaned Wealth Data
wealth_data_cleaned.head()
```
### Initial Analysis and Visualizations of Educational Attainment:
Initially, the focus was on magnet schools. The dataset was filtered to include only these schools, and then educational attainment was analyzed based on the prevalence of magnet schools in each county.

![image](https://github.com/anjanabegur/ENM3440HW3/assets/143770425/79e815f2-b61f-4b54-bc8c-e196e8a5198d)

Magnet Schools and Educational Attainment: This scatter plot shows the relationship between the presence of magnet schools (MAGNET_TEXT) and the percentage of educational attainment (estimate) in the county. Each point represents a school within a county.

Wealth and Educational Attainment: This scatter plot illustrates the relationship between the median income of a county and the percentage of educational attainment. Each point represents a school, with its position indicating the median income of the county it's located in and the educational attainment percentage of that county.

Code:
```
import matplotlib.pyplot as plt
import seaborn as sns

# Converting necessary columns to numeric for analysis
complete_merged_dataset['Median_Income'] = pd.to_numeric(complete_merged_dataset['Median_Income'], errors='coerce')
complete_merged_dataset['estimate'] = pd.to_numeric(complete_merged_dataset['estimate'], errors='coerce')

# Dropping rows with missing values in these columns for accurate analysis
analysis_dataset = complete_merged_dataset.dropna(subset=['Median_Income', 'estimate'])

# Visualizations
plt.figure(figsize=(16, 6))

# Visualization 1: Relationship between the presence of magnet schools and educational attainment
plt.subplot(1, 2, 1)
sns.scatterplot(data=analysis_dataset, x='MAGNET_TEXT', y='estimate')
plt.title('Magnet Schools and Educational Attainment')
plt.xlabel('Magnet School Presence')
plt.ylabel('Educational Attainment (%)')

# Visualization 2: Relationship between median income and educational attainment
plt.subplot(1, 2, 2)
sns.scatterplot(data=analysis_dataset, x='Median_Income', y='estimate')
plt.title('Wealth and Educational Attainment')
plt.xlabel('Median Income ($)')
plt.ylabel('Educational Attainment (%)')

plt.tight_layout()
plt.show()
```

### Linear Regression 
The regression analysis provided insights into the relationship between magnet school presence, wealth (median income), and educational attainment:
#### Model Summary:
#### R-squared: The R-squared value is 0.711, indicating that about 71.1% of the variability in educational attainment is explained by the model.

#### Coefficients:
Magnet School Presence (MAGNET_BINARY): The coefficient for magnet schools is -0.4440, but it's not statistically significant (p-value = 0.624). This implies that the presence of magnet schools, as per this model, does not have a significant impact on educational attainment when controlling for median income.

Median Income: The coefficient for median income is 0.3738 and is highly significant (p-value < 0.001). This suggests a positive correlation between the median income of a county and the educational attainment level, where higher income is associated with higher levels of educational attainment.

#### Statistical Significance:
The p-value for the magnet school variable suggests that it is not statistically significant in predicting educational attainment when controlling for median income.
The median income variable, however, is statistically significant, indicating a strong relationship with educational attainment.

#### Interpretation:
For each unit increase in median income (in thousands of dollars), there is an estimated increase of 0.3738% in the educational attainment level, holding magnet school presence constant.
The presence or absence of a magnet school does not significantly affect the educational attainment level when median income is accounted for in this model.
This analysis suggests that while wealth, as measured by median income, is a significant predictor of educational attainment in a county, the presence of magnet schools in and of itself may not be as influential, at least not in a way that is statistically discernible from this dataset and model.

Code:
```
import statsmodels.api as sm

# Preparing the data for regression analysis
# Converting MAGNET_TEXT to a binary variable for regression
analysis_dataset['MAGNET_BINARY'] = analysis_dataset['MAGNET_TEXT'].apply(lambda x: 1 if x == 'Yes' else 0)

# Independent variables: presence of magnet school and median income
X = analysis_dataset[['MAGNET_BINARY', 'Median_Income']]
# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Dependent variable: educational attainment
y = analysis_dataset['estimate']

# Running the regression model
model = sm.OLS(y, X).fit()

# Getting the summary of the regression
model_summary = model.summary()
model_summary

```
###  Logistic Regression Approach
As the original regression model did not provide significant results, I attempted a different analysis using a logistic regression model. 
#### Model Evaluation:
#### Classification Report:
Precision and Recall for Class 1 (High Attainment): The model failed to correctly predict any of the high attainment cases (class 1), as indicated by a precision and recall of 0.00. This suggests that the model is not effective in identifying high educational attainment based solely on magnet school presence.

Accuracy: The overall accuracy is 62%, but this is misleading as the model only predicts 'low' educational attainment (class 0).

Confusion Matrix: The matrix shows 95 true negatives (correctly predicted low attainment) but 57 false negatives (high attainment cases incorrectly predicted as low). The model did not predict any true positives (correct high attainment) or false positives.

#### Interpretation and Considerations
The logistic regression model, when only considering the presence of magnet schools, does not appear to be effective in predicting educational attainment. This could suggest that magnet school presence alone, without accounting for other factors like wealth or demographic variables, is not a strong predictor of high educational attainment in this dataset.
The lack of predictive power in this model might indicate the need for additional variables or a different analytical approach to better understand the relationship between attending a magnet school and educational attainment.

Code: 
```
# Logistic Regression Model without considering wealth (median income)
# Independent variable for logistic regression: magnet school presence only
X_logistic_magnet_only = analysis_dataset[['MAGNET_BINARY']]

# Splitting the dataset into training and testing sets for the new model
X_train_magnet, X_test_magnet, y_train_magnet, y_test_magnet = train_test_split(X_logistic_magnet_only, y_logistic, test_size=0.3, random_state=42)

# Logistic Regression Model for magnet school presence only
logistic_model_magnet_only = LogisticRegression()
logistic_model_magnet_only.fit(X_train_magnet, y_train_magnet)

# Predictions for magnet only model
y_pred_magnet = logistic_model_magnet_only.predict(X_test_magnet)

# Evaluation for magnet only model
logistic_report_magnet = classification_report(y_test_magnet, y_pred_magnet)
logistic_confusion_matrix_magnet = confusion_matrix(y_test_magnet, y_pred_magnet)

logistic_report_magnet, logistic_confusion_matrix_magnet

```

### Propensity Score Matching
Due to the lack of significant results with regression, I once again changed approaches, this time to a propensity score matching approach. I initially had ChatGPT do it one way, but then realized it wasn't taking into account the wealth confounder, so I made it redo the analysis. The revised version is outlined below. 

#### Steps:
Incorporate Wealth Data into Propensity Score Calculation: We'll merge the wealth data into our dataset and use it along with other characteristics to calculate the propensity scores.

Recalculate Propensity Scores: With the wealth data included, we'll recalculate the propensity scores for each school.

Perform Matching Again: We'll match magnet schools with non-magnet schools based on these new propensity scores.

Compare Educational Attainment: We'll compare the educational attainment between the newly matched schools, with the wealth factor accounted for in the matching.

Code:
```
# Merging the wealth data with the existing dataset for recalculating propensity scores
# We will use the median household income as the wealth indicator
# First, we need to prepare the wealth data for merging
wealth_data_prepared = wealth_data_cleaned.rename(columns={'FIPS': 'county_fips'}).dropna(subset=['Median_Income'])
wealth_data_prepared['Median_Income'] = pd.to_numeric(wealth_data_prepared['Median_Income'], errors='coerce')

# Merging the wealth data with the existing dataset
ccd_geocode_edu_wealth_merged = pd.merge(ccd_geocode_edu_reduced_all, wealth_data_prepared[['county_fips', 'Median_Income']], on='county_fips', how='left')

# Preparing independent variables for the new propensity score calculation
# Including FIPS code and median income
X_propensity_wealth = pd.get_dummies(ccd_geocode_edu_wealth_merged[['CNTY', 'Median_Income']], drop_first=True)

# Fitting the logistic regression model for calculating propensity scores with wealth data included
propensity_model_wealth = LogisticRegression(max_iter=1000)
propensity_model_wealth.fit(X_propensity_wealth, y_propensity_all)

# Predicting new propensity scores with wealth data included
propensity_scores_wealth = propensity_model_wealth.predict_proba(X_propensity_wealth)[:, 1]

# Adding new propensity scores to the dataset
ccd_geocode_edu_wealth_merged['propensity_score'] = propensity_scores_wealth

# Checking the first few rows of the dataset with new propensity scores
ccd_geocode_edu_wealth_merged.head()
```
We obtained the following visualizations from our results.
![image](https://github.com/anjanabegur/ENM3440HW3/assets/143770425/83478541-ca91-4f40-99a5-71cc6e3ec328)

#### Interpretation of Visualizations:
Both magnet and matched non-magnet schools exhibit similar distributions of educational attainment, as indicated by the overlapping histograms.
The boxplots for both groups also depict similar distributions, reinforcing the histogram findings.

#### Statistical Comparison:
Mean Educational Attainment: The mean educational attainment is identical for both magnet schools and matched non-magnet schools, approximately 30.69%.
Median Educational Attainment: The median educational attainment is the same for both groups, around 30.83%.

Code:
```
# Preparing data for comparison with wealth consideration
magnet_attainment_wealth = matched_schools_wealth['estimate']
matched_non_magnet_attainment_wealth = matched_schools_wealth['matched_non_magnet_estimate']

# Visualization: Comparing educational attainment between magnet and matched non-magnet schools with wealth consideration
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(magnet_attainment_wealth, bins=20, alpha=0.7, label='Magnet Schools')
plt.hist(matched_non_magnet_attainment_wealth, bins=20, alpha=0.7, label='Matched Non-Magnet Schools')
plt.title('Educational Attainment Comparison with Wealth Consideration')
plt.xlabel('Educational Attainment (%)')
plt.ylabel('Frequency')
plt.legend()

# Boxplot comparison with wealth consideration
plt.subplot(1, 2, 2)
data_to_plot_wealth = [magnet_attainment_wealth, matched_non_magnet_attainment_wealth]
plt.boxplot(data_to_plot_wealth, labels=['Magnet Schools', 'Matched Non-Magnet Schools'])
plt.title('Educational Attainment Comparison with Wealth Consideration')
plt.ylabel('Educational Attainment (%)')

plt.tight_layout()
plt.show()

# Statistical comparison with wealth consideration: Mean and median educational attainment
mean_magnet_wealth = magnet_attainment_wealth.mean()
mean_non_magnet_wealth = matched_non_magnet_attainment_wealth.mean()
median_magnet_wealth = magnet_attainment_wealth.median()
median_non_magnet_wealth = matched_non_magnet_attainment_wealth.median()

mean_magnet_wealth, mean_non_magnet_wealth, median_magnet_wealth, median_non_magnet_wealth
```
## Final Analyses and Additional Questions
Given that none of the statistical analyses yielded significant results, I wanted to generate some additional visualizations and ask a few final questions before writing the conclusion. I asked ChatGPT to do a quick analysis on what the relationship is between wealth and magnet school attendance. I obtained the plots below. 

![image](https://github.com/anjanabegur/ENM3440HW3/assets/143770425/e06217f2-2c1f-45e5-b442-56a16295abf8)

![image](https://github.com/anjanabegur/ENM3440HW3/assets/143770425/7396c5bc-aa5e-4317-a221-afe856b0702f)

### Density Plot of Educational Attainment:
The overlayed density plots illustrate the distribution of educational attainment for magnet and matched non-magnet schools. The similarity in their distributions reinforces our conclusion that there is no significant difference in educational attainment between the two types of schools when controlling for wealth.

### Probability of Attending a Magnet School Based on Wealth:

The scatter plot visualizes the probability of attending a magnet school based on county wealth. The plot shows varying probabilities across different levels of median income.

### Logistic Regression Coefficient:
The coefficient for median income in the logistic regression model is approximately 0.00078. This indicates a positive but very small relationship between county wealth and the likelihood of attending a magnet school. A higher median income in the county is associated with a slightly higher probability of a school being a magnet school.

## Conclusion
The comprehensive analysis revealed that the presence of magnet schools, when controlled for wealth and other factors, does not significantly impact educational attainment compared to non-magnet schools. The propensity score matching approach, accounting for socio-economic variables, showed similar educational outcomes for both types of schools. Additionally, logistic regression analysis indicated that wealthier counties have only a marginally higher likelihood of hosting magnet schools. These findings suggest that while magnet schools offer unique educational environments, their influence on educational attainment may be comparable to that of non-magnet schools when socio-economic factors are considered. This study highlights the complexity of educational dynamics and the importance of multifaceted analysis in understanding the role of specialized schools within the broader educational landscape.

## Notes on Question Modification Dialogue
As I was able to find data that was a relatively reasonable proxy for what Lizzy was initially asking, we didn't have too much in-depth discussion about my changing of her question. Initially, when I was still looking for data, we did discuss the extent to which it would be okay to modify the question, but as I found the datasets that I used, I realized I wouldn't have to modify it that much anyways.

## Personal Reflection
This was by far the hardest assignment I have had in this class so far. I usually look for interesting datasets and then formulate a question based on what I find, so starting out with a particular question was in and of itself a challenge, since I had to find very specific datasets to match what I was looking for. I think this is the reason I ended up having to use 4 datasets as well. The second major challenge I faced was actually because of the fact that I had so many datasets. Merging them through ChatGPT was a bit of a nightmare due to MemoryLoss errors, and after spending a few hours looking for data, this part of the project easily took at least another 4 hours. I had to go through quite a few iterations of each dataset to filter it down to sizes that were manageable for ChatGPT, and even after cutting them down significantly, I was still struggling with memory issues. I ended up taking a closer look at how exactly it was merging data, and realized that it was an issue with differing granularity amongst the county, school, state, and zip code identifiers that it was attempting to merge over. ChatGPT was creating datasets with over a million rows in an attempt to merge the CCD and GeoCode data, each of which only had about 10,000 rows. Solving this issue required some work on my end to optimize the key merging columns so that it would work without creating too many rows, but I ended up figuring out a way to make the first two merges into a one to one mapping so that the only additional rows were being added in the last step. After ChatGPT was finally able to merge all 4 datasets together (I felt immense joy in this moment) it was relatively smooth sailing, and I just had to go through several iterations of different types of analyses, none of which threw too many errors. Although the ultimate conclusion was that there is not a significant relationship between the two factors I was trying to investigate, I feel like I learned a lot about data processing through this assignment. 

