### Steps Taken for the Challenge
*(Documented in English to align with the instructions)*

#### Installation and Environment Setup
- **Objective:** Prepared the environment for executing the Jupyter Notebook code.
  - Created a `.venv` and installed all required dependencies.
  - Noted that XGBoost was not in any `requirements.txt` files, indicating it was a suggested model rather than a mandated one.
- **Fix:**
  - Manually installed XGBoost (`pip install xgboost`) and added it to `requirements.txt` for environment consistency.

#### Data Loading
- **Objective:** Loaded and inspected the dataset for analysis.
  - Dataset located in `data/` folder, loaded into a Pandas DataFrame from `data\data.csv`.
- **Results of `data.info()`:**
  - Dataset has 68,206 rows and 18 columns.
  - Some columns have null values (e.g., `Vlo-O` has one missing value).
  - Date columns (`Fecha-I`, `Fecha-O`) are in `object` format and require conversion to `datetime`.
- **Available Data Includes:**
  - Scheduled flight info: `Fecha-I`, `Vlo-I`, `Ori-I`, `Des-I`, `Emp-I`.
  - Operated flight info: `Fecha-O`, `Vlo-O`, `Ori-O`, `Des-O`, `Emp-O`.
  - Additional data: `DIA`, `MES`, `AÑO`, `DIANOM`, `TIPOVUELO`, `OPERA`, `SIGLAORI`, `SIGLADES`.


**Summary of exploration.ipynb:**

1. Enhanced a series of Python cells to improve clarity, compatibility, and efficiency for team collaboration.
2. Refactored sns.set(style="darkgrid") to work globally.

**Summary of "1. Data Analysis: First Sight" Cell Modifications**

#### Actions Performed:
1. **Corrected Missing `x` and `y` Parameters:**

2. **Added Concise Documentation:**
   - Applied to all cells to improve readability and team understanding:
     - **Airline Flights:** Counts flights by airline (`OPERA`).
     - **Day Flights:** Counts flights by day (`DIA`).
     - **Month Flights:** Counts flights by month (`MES`).
     - **Day of Week Flights:** Counts flights by day of week (`DIANOM`).
     - **Flight Type Flights:** Counts flights by type (`TIPOVUELO`).
     - **Destination Flights:** Counts flights by destination (`SIGLADES`).
   - Comments focus on objectives (e.g., "Count flights by X")

3. **Refactored for Efficiency:**
   - **Cell: Flights by Day of Week**
     - Replaced manual index-based reordering with `reindex(day_order, fill_value=0)` using a predefined `day_order` list (Monday to Sunday).
     - Improved efficiency and robustness by avoiding hardcoded indices and ensuring all days are included, even if missing from data.

#### Outcome:
- All cells now use consistent, version-safe Seaborn syntax.
- Documentation enhances team collaboration by clarifying each cell’s purpose and settings.
- The refactored "Day of Week" cell is more maintainable and adaptable to varying datasets.


**Summary of "2. Features Generation" Cell Modifications**

1. Added documentation to enhance team collaboration by clarifying each cell’s purpose and settings.

**Summary of "3. Data Analysis: Second Sight" Cell Modifications**

#### Actions Performed:
1. Added Concise Documentation to all cells to improve readability and team understanding:

2. **Refactored for Efficiency and Logical Ordering:**
   - **Cell: Delay Rate by Destination**
     - Ensured `destination_rate_values` and `destination_rate['Tasa (%)']` are aligned.
     - Sorted the values of `destination_rate_values` according to the values of `destination_rate['Tasa (%)']` to display the graph in a logical order.

   - **Cell: Delay Rate by Airline**
     - Ensured `airlines_rate_values` and `airlines_rate['Tasa (%)']` are aligned.
     - Sorted the values of `airlines_rate_values` according to the values of `airlines_rate['Tasa (%)']` to display the graph in a logical order.

#### Outcome:
- All cells now have clear documentation explaining their purpose and functionality.
- The refactored cells ensure that graphs are sorted in a logical order, enhancing readability and interpretability.
- Enhanced team collaboration by providing clear and concise comments for each cell, focusing on the importance of logical ordering in visualizations.

#### Suggestions:

Refactor of get_rate_from_column foundational utility function to the following one:

def get_rate_from_column(data, column):
    """Calculate delay rates (%) for each unique value in the specified column."""
    # Count delays (where delay == 1) per unique value in the column
    delays = data[data['delay'] == 1][column].value_counts()
    
    # Get total counts per unique value in the column
    total = data[column].value_counts()
    
    # Compute rates: total / delays, fill with 0 where no delays, round to 2 decimals
    rates = (total / delays).fillna(0).round(2)
    
    # Return as DataFrame with 'Tasa (%)' column
    return pd.DataFrame(rates, columns=['Tasa (%)'])

The original uses a slow Python loop with iterrows() to count delays, which is inefficient for large DataFrames. The refactored version uses value_counts() on a filtered DataFrame (data[data['delay'] == 1]), leveraging Pandas’ optimized C-based operations.

The original was keept for the sake of the exploration.ipynb execution.


**Summary of "4. Model Selection" Cell Modifications**

1. Added documentation to enhance team collaboration by clarifying each cell’s purpose and settings.

**Summary of "5. Data Analysis: Third Sight" Cell Modifications**

1. Added documentation to enhance team collaboration by clarifying each cell’s purpose and settings.

**Summary of "## 6. Training with Improvement" Cell Modifications**

1. Added documentation to enhance team collaboration by clarifying each cell’s purpose and settings.

**Summary of "## 7. Data Science Conclusions"**

# ### Model Selection and Justification

# #### Evaluation of Models

# 1. **XGBoost with Feature Importance and Class Balance**:
   - **Performance**: This model shows a balanced performance with good precision and recall for both classes. The F1-score is also high, indicating a good trade-off between precision and recall.
   - **Class Balance**: The model incorporates class balancing techniques, which improves the recall for the minority class (class "1").
   - **Feature Importance**: Using the top 10 features does not degrade the model's performance, making it more efficient.

# 2. **Logistic Regression with Feature Importance and Class Balance**:
   - **Performance**: This model also shows good performance metrics, but it may not handle class imbalance as effectively as XGBoost.
   - **Class Balance**: The model uses class weights to balance the classes, but the improvement in recall for the minority class is not as significant as in XGBoost.
   - **Feature Importance**: Similar to XGBoost, using the top 10 features does not degrade the model's performance.

# ### Conclusion

 Based on the evaluation metrics and considerations, the **XGBoost model with feature importance and class balance** is the best choice for the following reasons:
 - **Balanced Performance**: It maintains a good balance between precision and recall for both classes.
 - **Effective Class Balancing**: The model's use of `scale_pos_weight` significantly improves the recall for the minority class, which is crucial for detecting delays.
 - **Efficient Feature Selection**: Using the top 10 features does not degrade the model's performance, making it more efficient without sacrificing accuracy.

# That is why the XGBoost model with feature importance and class balance is the most suitable model for this problem.

**Summary of "8. API Development (`api.py`)"**

The API was implemented using **FastAPI**, allowing users to send flight data and receive delay predictions.

### **Endpoints**
- `GET /health` – Health check to confirm the API is running.
- `POST /predict` – Accepts flight data and returns a binary prediction (`0` = no delay, `1` = delayed).

### **Training Strategy**
Instead of training the model on every request, it is **trained once when the API starts**. This ensures:
- Faster response times.  
- The model remains in memory for reuse.  
- Consistency across predictions.

#### **Training Implementation at API Startup to keep the structure of the challenge**

# Initialize and train the model at server start
model = DelayModel()
data = pd.read_csv("/data/data.csv")  # Ensure dataset is available
X, y = model.preprocess(data, target_column="delay")
model.fit(X, y)  # Train the model once

**## Summary of "9. Deployment of the API"**

The trained model and API were successfully containerized and deployed using **Google Cloud Run**.

### **Containerization and Deployment**
1. **Dockerfile Creation**  
   - A `Dockerfile` was prepared with the necessary dependencies.
   - Exposed port **8080**, aligning with Cloud Run's default expected port.
   - Used `uvicorn` to run the FastAPI application.

2. **Building and Pushing the Docker Image**  
   - The image was built locally and tagged:
     ```sh
     docker build -t gcr.io/latam-challenge-mle-452917/flight-delay-api .
     ```
   - It was then pushed to **Google Container Registry (GCR)**:
     ```sh
     docker push gcr.io/latam-challenge-mle-452917/flight-delay-api
     ```

3. **Deploying the API to Google Cloud Run**  
   - The API was deployed using the following command:
     ```sh
     gcloud run deploy flight-delay-api \
       --image gcr.io/latam-challenge-mle-452917/flight-delay-api \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
     ```
   - The service was successfully deployed at:  
     **`https://flight-delay-api-48997287020.us-central1.run.app`**

4. **Updating the Makefile**  
   - The API’s URL was updated in the **`Makefile`** (`line 26`):
     ```make
     STRESS_URL = https://flight-delay-api-48997287020.us-central1.run.app
     ```

5. **Configuring Cloud Run Autoscaling**  
   - Initially, errors were encountered due to a lack of available instances.
   - To ensure availability, **minimum instances** were set to `1`, and **maximum instances** were capped at `10`:
     ```sh
     gcloud run services update flight-delay-api \
       --region=us-central1 \
       --min-instances=1 \
       --max-instances=10 \
       --cpu=1 \
       --memory=1Gi \
       --timeout=300s
     ```

### **Stress Testing**
- The deployed API underwent a **stress test** using **Locust**, simulating **100 concurrent users**.
- **Test execution:**
  make stress-test
    Results Summary:
    Total requests: 1,577
    Failure rate: 0.00% (No failed requests)
    Median response time: ~1,100ms
    99th percentile latency: ~2,300ms
    Max latency observed: ~3,211ms

**## Summary of "10. - CI/CD Implementation"** 

To ensure a proper **CI/CD (Continuous Integration / Continuous Deployment)** implementation for this development, we set up GitHub Actions workflows.

### **CI (Continuous Integration)**
The **CI workflow (`ci.yml`)** is responsible for:
- **Linting and code quality checks** to ensure code follows best practices.
- **Running unit tests** to verify that changes do not introduce new errors.
- **Checking dependencies** to prevent missing or incompatible packages.

#### **Steps Taken**
1. **Created a new folder `.github/workflows/`** in the repository.
2. **Copied the provided `ci.yml` file** into this directory.
3. **Configured the CI pipeline** to:
   - Install dependencies from `requirements.txt` and `requirements-test.txt`.
   - Run tests using `pytest` with code coverage reports.
   - Validate Python syntax and style.

### **CD (Continuous Deployment)**
The **CD workflow (`cd.yml`)** is designed to:
- **Verify that deployment files exist** (`Dockerfile`, `Makefile`).
- **Check Google Cloud Run deployment settings** (service name and region).
- **Ensure the setup is correct for future deployments**.

#### **Steps Taken**
1. **Created the `cd.yml` workflow** inside `.github/workflows/`.
2. **Adjusted the configuration** to:
   - Validate that essential deployment files (`Dockerfile`, `Makefile`) are present.
   - Confirm Cloud Run deployment settings (expected service name and region).
   - Skip actual deployments to **avoid pushing changes automatically**.

> **Note:** Since this challenge is for review purposes, the CD pipeline **does not deploy to Google Cloud Run** but ensures the correct setup.
