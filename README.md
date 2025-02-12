# **Medical Insurance Cost Prediction Using Linear Regression**  

## **Overview**  
This project utilizes **Linear Regression** to predict **medical insurance costs** based on various factors such as age, BMI, smoking status, and region. The dataset used in this project is `insurance.csv`, which contains relevant attributes affecting insurance charges.  

## **Dataset Features**  
The dataset includes the following columns:  

- **age**: Age of the policyholder  
- **sex**: Gender (male/female)  
- **bmi**: Body Mass Index (BMI)  
- **children**: Number of children covered by health insurance  
- **smoker**: Whether the policyholder is a smoker (yes/no)  
- **region**: Residential region (southeast, southwest, northeast, northwest)  
- **charges**: The medical insurance cost (target variable)  

---

## **Project Workflow**  

### **1. Data Loading and Exploration**  
- The dataset is loaded using **pandas** and basic exploratory data analysis (EDA) is performed.  
- We check the number of rows and columns, missing values, and statistical insights.  

### **2. Data Visualization**  
We use **Seaborn** and **Matplotlib** to visualize the distribution of key features, such as:  
- Age distribution  
- BMI distribution  
- Number of children per policyholder  
- Smoking status distribution  
- Regional distribution  
- Insurance charges distribution  

### **3. Data Preprocessing**  
- **Encoding categorical variables**:  
  - `sex` → Male (0), Female (1)  
  - `smoker` → Yes (0), No (1)  
  - `region` → Southeast (0), Southwest (1), Northeast (2), Northwest (3)  
- **Feature-target split**:  
  - Features (`X`): All columns except `charges`  
  - Target (`Y`): `charges`  

### **4. Model Training (Linear Regression)**  
- The dataset is split into **training (80%)** and **testing (20%)** sets using `train_test_split()`.  
- A **Linear Regression model** is trained using `sklearn.linear_model.LinearRegression()`.  

### **5. Model Evaluation**  
- Predictions are made on both **training** and **testing** data.  
- Performance is measured using **R² (R-squared) score**, which indicates how well the model explains the variance in medical charges.  

### **6. Making Predictions**  
- A predictive system is built where a sample input (e.g., `input_data = (56,0,40.3,0,1,1)`) is passed to the model.  
- The model predicts the expected medical insurance cost based on the given input.  

---

## **Results & Insights**  
- The model provides an R² score to indicate prediction accuracy.  
- Higher BMI and smoking status significantly impact insurance costs.  
- The model can be further improved using more advanced techniques like **Polynomial Regression** or **Feature Engineering**.  

---

## **Future Improvements**  
- Use more complex regression models such as **Random Forest Regressor** or **XGBoost**.  
- Implement **Hyperparameter Tuning** for better model accuracy.  
- Deploy the model using **Flask or FastAPI** for real-world applications.  

---

**Basil Siddiqui**  
