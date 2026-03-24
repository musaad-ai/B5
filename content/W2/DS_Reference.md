# Data Analytics

See [Data Analytics](https://www.coursera.org/articles/data-analytics) (a more general term than: Data Analysis).

## 1. Descriptive Analytics

Asks: **What happened?** This is the foundation of business reporting. It uses historical data to summarize performance.

Examples:

- A retail monthly sales report showing which regions met their quotas.
- Screen Time: Your phone provides a weekly report showing you spent 5 hours on social media and 2 hours on productivity apps.
- A social media dashboard summarizing total likes, shares, and mentions from the past week.

## 2. Diagnostic Analytics

Asks: **Why did it happen?** This involves "drilling down" into the data to find dependencies and causes.

Examples:

- Production Defects: Correlating a sudden spike in product returns to a single batch of faulty raw materials from a new supplier.

## 3. Predictive Analytics

Asks: **What will happen?** This uses statistical models and machine learning to forecast future trends.

Examples:

- Demand Forecasting: A manufacturer uses historical orders and economic indicators to estimate how many units of a product they will need to produce for the holiday season.
- Keyboard Autocomplete: Your phone predicts and suggests the next three words of your sentence based on your typing habits.

## 4. Prescriptive Analytics

Asks: **How can we make it happen?** The most advanced stage, where the analysis recommends specific actions to achieve an optimal outcome.

Examples:

- Dynamic Inventory Management: An e-commerce platform automatically triggers discount codes for overstocked items while simultaneously increasing the price of low-stock, high-demand goods to maximize margin.
- An airline's pricing algorithm automatically adjusting ticket costs in real-time based on demand and weather patterns.
- Sleep Coaching: A fitness wearable analyzes your high stress levels and tells you to go to bed 30 minutes earlier tonight to ensure full recovery.

## Key Steps in Data Analytics

### 1.1 Data Loading

Gather relevant data from sources:

1. Files (csv, excel, txt, json, xml, pdf, etc.)
2. Surveys (Google Forms, SurveyMonkey)
3. Web Scraping (BeautifulSoup, Scrapy, Selenium)
4. APIs (Twitter, Facebook, Google Maps, etc.)
5. Databases (SQL, NoSQL)

### 1.2 Data Cleaning

Prepare data for analysis by:

1. Select relevant columns
2. Impute missing values (or flag them as such)
    - Mean, Median, Mode
    - Forward fill, Backward fill
    - Interpolation
3. Remove duplicates
4. Correct errors (invalid values)
5. Remove irrelevant and redundant features
6. Column names' normalization
7. Value formats:
    1. Convert data types (e.g., `str -> datetime`)
    2. inconsistencies (e.g., `Female` and `F` both present in `gender` column)
    3. One format for dates, phone numbers, etc.

### 2.1 Scaling & Outlier Treatment

1. Scale the data:
    1. Normalize
        - Log Transformation
    2. Standardize
        - Z-score
        - Min-max Scaling
        - Robust Scaling
2. Handle Outliers
    - Z-score method
    - IQR method
    - 95th percentile
    - 99th percentile
    - Domain knowledge
3. Handle data imbalance
4. Encode categorical variables

### 2.2 Feature Engineering

Based on domain knowledge and questions to be answered:

1. Create new variables
    1. `age` from `date_of_birth`
    2. `BMI` from `weight` and `height`
    3. `season` from `date`
2. *Data Enrichment* - Add new data from external sources
    1. Geocoding (convert address to latitude and longitude using **Google Maps API**)
    2. Sentiment analysis (analyze text data using **OpenAI API**)
3. *Binning*
    1. `age_group` from `age`
    2. `income_group` from `income`
    3. `weight_category` from `weight`
4. Handle date-time variables
    1. Extract year, month, day, day of week, etc.
    2. Time since last purchase
    3. Time since first visit
5. Aggregate data
    1. `total_sales` from `sales` table
    2. `orders_count` from `orders` table
    3. `maximum_amount` from `transactions` table

### 2.3 Feature Selection

1. Select Features
    1. Correlation Coefficient
    2. Mutual information

### 3.1 Uni-variate Analysis

Understand the distribution of each variable.

1. *Descriptive Statistics*
    - Measure of central tendency (mean, median, mode)
    - Dispersion (range, variance, standard deviation)
    - Shape (skewness, kurtosis).
2. *Visualizations*
    - Histogram
    - Boxplot
    - Density plot
    - Violin plot
    - Bar plot
    - Pie chart
    - Frequency table
    - Word cloud

### 3.2 Multi-variate Analysis

Understand the relationship between multiple variables.

1. *Descriptive Statistics*
   - Covariance
   - Correlation
2. *Visualizations*
   - Scatter plot
   - Line plot
   - Heatmap
   - Pairplot
   - Boxplot
   - Violin plot
   - Bar plot
   - Stacked bar plot
   - Grouped bar plot


# Python Packages for Data Analysis

Python tools for data science are built on top of the following fundamental packages/libraries:

- **NumPy**: The fundamental **package** for scientific computing with Python.
- **SciPy**: Fundamental **algorithms** for scientific computing in Python.
- **Matplotlib** is a comprehensive library for creating static, animated, and interactive **visualizations** in Python.
Such Python libraries use **C** underneath to achieve high performance, yet provides it in a simple Pythonic Interface / API. As shown in the image below (for NumPy):

![Numpy Languages](../assets/numpy-languages.png)
We will use the following libraries in this course:

1. **Pandas**: for data wrangling and analysis.
2. **Seaborn**: for data visualization.
3. **statsmodels**: for statistical modeling and inference.
