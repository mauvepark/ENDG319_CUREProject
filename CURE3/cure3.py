import pandas as pd
import matplotlib.pyplot as plt

# load csv
data = pd.read_csv("cure3.csv")

# dataframe
df = data[["age", "weight", "height", "sleep", "bmi", "exercise" , "sugar_intake", "smoking", "alcohol", "married", "profession", "health_risk"]]

# separate classes
low = df[df["health_risk"].str.lower() == "low"]
high = df[df["health_risk"].str.lower() == "high"]

# figures
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# plot1: bmi vs health risk histogram
axes[0].hist(low["bmi"], bins=20, histtype="step", color="blue", label="Low Risk", linewidth=1.5)
axes[0].hist(high["bmi"], bins=20, histtype="step", color="orange", label="High Risk", linewidth=1.5)
axes[0].set_xlabel("BMI", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].set_title("BMI Distribution by Health Risk", fontsize=12)
axes[0].legend(frameon=True, fontsize=10)

# plot2: age vs bmi scatter
axes[1].scatter(low["age"], low["bmi"], color="blue", s=8, alpha=0.6, label="Low Risk", edgecolor="none")
axes[1].scatter(high["age"], high["bmi"], color="orange", s=8, alpha=0.6, label="High Risk", edgecolor="none")
axes[1].set_xlabel("Age (years)", fontsize=11)
axes[1].set_ylabel("BMI", fontsize=11)
axes[1].set_title("Age vs BMI", fontsize=12)
axes[1].legend(frameon=True, fontsize=10)

# plot3: weight vs bmi scatter
axes[2].scatter(low["weight"], low["bmi"], color="blue", s=8, alpha=0.6, label="Low Risk", edgecolor="none")
axes[2].scatter(high["weight"], high["bmi"], color="orange", s=8, alpha=0.6, label="High Risk", edgecolor="none")
axes[2].set_xlabel("Weight (kg)", fontsize=11)
axes[2].set_ylabel("BMI", fontsize=11)
axes[2].set_title("Weight vs BMI", fontsize=12)
axes[2].legend(frameon=True, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0)
plt.show()

