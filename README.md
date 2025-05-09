# 📘 Basic Maths & Statistics for Machine Learning (Beginner-Friendly)

This project introduces fundamental mathematical and statistical concepts commonly used in Machine Learning, with on going Python examples and visualisations. It's perfect for beginners looking to build a strong foundation before diving into ML models.

This approach is quite direct and simplistic, which I feel is a great way to grasp these concepts. This is as opposed to the lengthy and overcomplicated explanations you'll find out there!

---

## 📂 Files Included

### `BasicMathsForMl.py` — *Basic Statistics*
Introduces core statistical tools used in data analysis and preprocessing:
- **Mean** – average value
- **Median** – middle value in a sorted list
- **Mode** – most frequent value (`scipy.stats.mode`)
- **Standard Deviation** – how spread out values are
- **Variance** – square of standard deviation
- **Percentiles** – useful for understanding data distribution

---

### `BiggerData.py` — *Larger Datasets & Distributions*
Demonstrates working with larger, randomly generated datasets:
- **Uniform distribution** using `np.random.uniform`
- **Histograms** for visualising distribution
- **Normal distribution** using `np.random.normal`
- **Scatter plots** to explore relationships between variables

---

### `Regression.py` — *Linear and Polynomial Regression*
Introduces regression models to explore relationships and make predictions:

#### 🔹 Linear Regression
- Uses `scipy.stats.linregress` to fit a line
- Outputs:
  - **Slope** and **intercept**
  - **Correlation coefficient (`r`)**
  - **P-value** and **standard error**
- Visualises the line of best fit over real data

#### 🔹 Polynomial Regression
- Fits curves using `np.polyfit` and `np.poly1d`
- Measures fit using **R² score** (`sklearn.metrics.r2_score`)
- Great for when data doesn't follow a straight line pattern

---

### `ScaleFeatures.py` — *Scaling features*
Demonstrating Scaling data into new values and the importance of doing so.

- Uses `StandardScaler()` to scale data using the standardisation formula. Z = x - u / s

---

### `KNN.py` — *K Nearest Neighbors*
- Uses `KNeighborsClassifier` to classify customer types from telecom data  
- Scales features using `StandardScaler()` for fair distance comparison  
- Evaluates accuracy for values of **k from 1 to 100**  
- Plots model accuracy and standard deviation to find the optimal `k`
- Predicts the majority vote of the k nearest training points
