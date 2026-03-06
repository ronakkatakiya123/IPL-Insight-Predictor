# IPL Data Analysis & Match Winner Prediction

This project focuses on analyzing historical Indian Premier League (IPL) cricket match data to derive insights and build a machine learning model to predict match winners based on target runs.

## 📊 Project Overview
The core of this project is to process and visualize data from IPL matches (2008–2024) to understand team performance, player statistics, and match outcomes. [cite_start]We implement a custom K-Nearest Neighbor (KNN) classifier to predict the winner of a match between two specific teams based on the target runs[cite: 326, 366].

## 📂 Datasets
[cite_start]This project uses two primary datasets[cite: 4, 317]:
* **`matches.csv`**: Contains match-level information (e.g., match ID, season, city, venue, teams, toss decisions, and winners).
* **`deliveries.csv`**: Contains ball-by-ball information (e.g., batting team, bowling team, batter, bowler, runs scored, and dismissal details).

## 🛠 Methodology

### 1. Data Cleaning
* [cite_start]**Extras**: Missing values in `extras_type` were filled with 0[cite: 120].
* [cite_start]**Winners**: Missing `winner` values were replaced with "No result"[cite: 13].
* [cite_start]**Dismissals**: Missing `player_dismissed` and `dismissal_kind` were filled with "Not out"[cite: 193].
* [cite_start]**Fielding**: Missing `fielder` values were categorized as "Not cought & Not run out"[cite: 193].

### 2. Exploratory Data Analysis (EDA)
* [cite_start]**Team Performance**: Analyzed the total number of matches played by each team[cite: 195].
* [cite_start]**Winner Distribution**: Visualized the distribution of match winners using pie charts[cite: 225].
* [cite_start]**Top Players**: Identified top run-scorers and top wicket-takers via grouping functions[cite: 285, 290].
* [cite_start]**Win/Loss by Season**: Extracted and visualized the winner of each season[cite: 252, 256].

### 3. Machine Learning (KNN Classifier)
* [cite_start]**Algorithm**: Implemented a custom K-Nearest Neighbor (KNN) algorithm using Euclidean distance[cite: 326, 331].
* [cite_start]**Prediction**: The model accepts input team names and a target run score to predict the likely winner[cite: 367].
* [cite_start]**Evaluation**: The model performance was evaluated using `accuracy_score`, `precision_score`, and `recall_score`[cite: 384].

## 🚀 How to Run
1.  **Clone the repository**:
    ```bash
    git clone [your-repository-url]
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
3.  **Run the analysis**: Open the main Jupyter Notebook provided in the repository to execute the data processing, visualization, and prediction cells sequentially[cite: 312, 321, 372].

## 📈 Results
The KNN classifier demonstrates the ability to predict match outcomes based on historical target run patterns. [cite_start]The model evaluation on the test set achieved an accuracy of 0.75[cite: 394].

---
*Developed as part of data analysis studies.*
