# ============================================================
# IPL MATCH ANALYSIS AND WINNER PREDICTION USING KNN
# ============================================================

# =========================
# 1. Import Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# =========================
# 2. Load Dataset
# =========================

deliveries = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

print("Deliveries Dataset Info")
deliveries.info()

print("\nMatches Dataset Info")
matches.info()


# =========================
# 3. Copy Dataset
# =========================

data = deliveries.copy()
d = matches.copy()


# =========================
# 4. Data Cleaning
# =========================

# Replace missing winners
d.loc[d["winner"].isna(), "winner"] = "No result"

# Handle missing values in deliveries dataset
data.loc[data["extras_type"].isna(), "extras_type"] = 0
data.loc[data["player_dismissed"].isna(), "player_dismissed"] = "Not out"
data.loc[data["dismissal_kind"].isna(), "dismissal_kind"] = "Not out"
data.loc[data["fielder"].isna(), "fielder"] = "Not caught & Not run out"

print("\nAfter Cleaning:")
data.info()
d.info()


# =========================
# 5. Team Participation Analysis
# =========================

team_matches = (d["team1"].value_counts() + d["team2"].value_counts()).sort_values(ascending=False)

print("\nMatches played by teams:")
print(team_matches)

team_matches.plot(kind="bar", figsize=(10,6))
plt.title("Matches Played by Teams")
plt.xlabel("Team")
plt.ylabel("Matches")
plt.show()


# =========================
# 6. Match Winners Distribution
# =========================

d["winner"].value_counts().head(12).plot(kind="pie", autopct="%.1f%%")
plt.title("Match Winners Distribution")
plt.ylabel("")
plt.show()


# =========================
# 7. IPL Winners by Season
# =========================

season_winners = d.drop_duplicates(subset=["season"], keep="last")[["season","winner"]]

print("\nSeason Winners:")
print(season_winners)

season_winners["winner"].value_counts().plot(kind="barh")
plt.title("IPL Titles by Team")
plt.xlabel("Number of Titles")
plt.show()


# =========================
# 8. Toss Decision Analysis
# =========================

d["toss_decision"].value_counts().plot(kind="pie", explode=[0,0.1])
plt.title("Toss Decision Distribution")
plt.ylabel("")
plt.show()


# =========================
# 9. Run Rate Calculation
# =========================

run_rate = (data["batsman_runs"].sum() / data.shape[0]) * 6
print("\nAverage Run Rate:", run_rate)


# =========================
# 10. Wicket Average
# =========================

wicket_avg = data.shape[0] / data["is_wicket"].sum()
print("Wicket Average:", wicket_avg)


# =========================
# 11. Top Run Scorers
# =========================

top_batters = data.groupby("batter")["batsman_runs"].sum().sort_values(ascending=False).head(7)

print("\nTop Run Scorers:")
print(top_batters)


# =========================
# 12. Top Wicket Takers
# =========================

top_bowlers = data.groupby("bowler")["is_wicket"].sum().sort_values(ascending=False).head(7)

print("\nTop Wicket Takers:")
print(top_bowlers)


# =========================
# 13. Team Wicket Analysis
# =========================

team_wickets = data.groupby("bowling_team")["is_wicket"].sum().sort_values(ascending=False).head(10)

print("\nTop Teams by Wickets:")
print(team_wickets)


# =========================
# 14. Team Run Analysis
# =========================

team_runs = data.groupby("batting_team")["batsman_runs"].sum().sort_values(ascending=False).head(10)

print("\nTop Teams by Runs:")
print(team_runs)


# =========================
# 15. Frequency of Runs
# =========================

def times_run(num):

    mask = data["batsman_runs"] == num
    d2 = data[mask]

    return d2.groupby("batter")["batsman_runs"].count().sort_values(ascending=False).head(10)

print("\nPlayers with most 4s:")
print(times_run(4))


# =========================
# 16. First Innings Runs
# =========================

first_inning = data[data["inning"] == 1]

first_inning_runs = first_inning.groupby("batting_team")["batsman_runs"].sum().sort_values(ascending=False)

print("\nFirst Innings Runs by Team:")
print(first_inning_runs)


# ============================================================
# MACHINE LEARNING PART
# ============================================================


# =========================
# 17. Euclidean Distance
# =========================

def euclidean_distance(x1, x2):

    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


# =========================
# 18. KNN Class
# =========================

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]


# =========================
# 19. Filter Matches Between Teams
# =========================

def teams(b1, b2):

    mask1 = d["team1"] == b1
    mask2 = d["team2"] == b2

    mask3 = d["toss_winner"] == b1
    mask4 = d["toss_decision"] == "bat"

    mask5 = d["toss_winner"] == b2
    mask6 = d["toss_decision"] == "field"

    m1 = d["team1"] == b2
    m2 = d["team2"] == b1

    return d[((mask1 & mask2) | (m1 & m2)) & ((mask3 & mask4) | (mask5 & mask6))]


# =========================
# 20. Winner Prediction
# =========================

t1 = input("\nEnter Batting Team: ")
t2 = input("Enter Bowling Team: ")

new = teams(t1, t2)

run = new["target_runs"]
win = new["winner"]

input_run = int(input("Enter Target Runs: "))

clf = KNN(k=3)

clf.fit(run.values.reshape(-1,1), win)

prediction = clf.predict([[input_run]])

print("\nPredicted Winner:", prediction)


# =========================
# 21. Train Test Split
# =========================

shuf = np.random.permutation(run.shape[0])

ratio = 0.2
n = int(np.floor((1-ratio) * run.shape[0]))

run_train = run.iloc[shuf[:n]]
win_train = win.iloc[shuf[:n]]

run_test = run.iloc[shuf[n:]]
win_test = win.iloc[shuf[n:]]


# =========================
# 22. Model Evaluation
# =========================

clf = KNN(k=3)

clf.fit(run_train.values.reshape(-1,1), win_train)

predictions = clf.predict(run_test.values.reshape(-1,1))

accuracy = accuracy_score(win_test, predictions)
precision = precision_score(win_test, predictions, average='weighted')
recall = recall_score(win_test, predictions, average='weighted')

print("\nModel Evaluation")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\nClassification Report")
print(classification_report(win_test, predictions))
