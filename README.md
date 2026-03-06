# IPL Insight Predictor

A comprehensive data analysis project exploring Indian Premier League (IPL) match and delivery statistics. This repository features data cleaning workflows, visualization of key team and player performance metrics, and a custom K-Nearest Neighbor (KNN) machine learning implementation to predict match winners based on target run data.

## 📊 Project Overview
The core of this project is to process and visualize data from IPL matches (2008–2024) to understand team performance, player statistics, and match outcomes. We implement a custom K-Nearest Neighbor (KNN) classifier to predict the winner of a match between two specific teams based on the target runs.

## 📂 Datasets
This project uses two primary datasets:
* **`matches.csv`**: Contains match-level information (e.g., match ID, season, city, venue, teams, toss decisions, and winners).
* **`deliveries.csv`**: Contains ball-by-ball information (e.g., batting team, bowling team, batter, bowler, runs scored, and dismissal details).

## 🛠 Methodology

### 1. Data Cleaning
* **Extras**: Missing values in `extras_type` were filled with 0.
* **Winners**: Missing `winner` values were replaced with "No result".
* **Dismissals**: Missing `player_dismissed` and `dismissal_kind` were filled with "Not out".
* **Fielding**: Missing `fielder` values were categorized as "Not cought & Not run out".

### 2. Exploratory Data Analysis (EDA)
* **Team Performance**: Analyzed the total number of matches played by each team.
* **Winner Distribution**: Visualized the distribution of match winners using pie charts.
* **Top Players**: Identified top run-scorers and top wicket-takers via grouping functions.
* **Win/Loss by Season**: Extracted and visualized the winner of each season.

### 3. Machine Learning (KNN Classifier)

* **Algorithm**: Implemented a custom K-Nearest Neighbor (KNN) algorithm using Euclidean distance.
* **Prediction**: The model accepts input team names and a target run score to predict the likely winner.
* **Evaluation**: The model performance was evaluated using `accuracy_score`, `precision_score`, and `recall_score`.

## 🚀 How to Run
1.  **Clone the repository**:
    ```bash
    git clone [your-repository-url]
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
3.  **Run the analysis**: Open the main Jupyter Notebook provided in the repository to execute the data processing, visualization, and prediction cells sequentially.

## 📈 Results
The KNN classifier demonstrates the ability to predict match outcomes based on historical target run patterns. The model evaluation on the test set achieved an accuracy of 0.75.

---
*Author: Ronak Katakiya*
