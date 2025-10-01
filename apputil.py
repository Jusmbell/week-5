"""Utility functions for Titanic weekly exercises.

Implements the required data transformation and visualization helpers:
 - survival_demographics / visualize_demographic
 - family_groups / last_names / visualize_families
 - determine_age_division / visualize_age_division (bonus)

All functions load the Titanic dataset from the provided GitHub URL. A small
cache is used so the CSV is only downloaded once per session.
"""

from functools import lru_cache
import plotly.express as px
import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"


@lru_cache(maxsize=1)
def _load_titanic() -> pd.DataFrame:
	"""Load and cache the Titanic dataset."""
	return pd.read_csv(TITANIC_URL)


def survival_demographics() -> pd.DataFrame:
	"""Return survival statistics by passenger class, sex, and age group.

	Age groups (inclusive bounds):
		- Child (0–12)
		- Teen (13–19)
		- Adult (20–59)
		- Senior (60+)

	Returns a DataFrame with all combinations of Pclass, Sex, and age_group
	(even if zero passengers fall into a bucket) and columns:
		n_passengers, n_survivors, survival_rate.
	"""

	df = _load_titanic().copy()

	# Define age groups
	bins = [0, 12, 19, 59, 120]
	labels = ["Child", "Teen", "Adult", "Senior"]
	df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

	# Prepare full index (cartesian product) to ensure all combinations appear
	pclasses = sorted(df["Pclass"].dropna().unique())
	sexes = sorted(df["Sex"].dropna().unique())
	age_cats = labels  # already ordered
	full_index = pd.MultiIndex.from_product([pclasses, sexes, age_cats], names=["Pclass", "Sex", "age_group"])

	grouped = (
		df.dropna(subset=["age_group"])  # exclude rows where age was NaN (no group)
		.groupby(["Pclass", "Sex", "age_group"], observed=False)
		.agg(n_passengers=("Survived", "count"), n_survivors=("Survived", "sum"))
		.reindex(full_index, fill_value=0)
		.reset_index()
	)
	grouped["survival_rate"] = grouped.apply(
		lambda r: 0 if r.n_passengers == 0 else r.n_survivors / r.n_passengers, axis=1
	)

	# Sort in a readable order
	grouped = grouped.sort_values(["Pclass", "Sex", "age_group"]).reset_index(drop=True)
	return grouped


def visualize_demographic():
	"""Create a Plotly figure showing survival rate by age group, sex, and class.

	Question addressed (also written in app):
		How does survival rate vary across passenger class and sex for each age group?
	"""

	data = survival_demographics()
	fig = px.bar(
		data,
		x="age_group",
		y="survival_rate",
		color="Sex",
		barmode="group",
		facet_col="Pclass",
		category_orders={"age_group": ["Child", "Teen", "Adult", "Senior"]},
		labels={"age_group": "Age Group", "survival_rate": "Survival Rate"},
		title="Survival Rate by Age Group, Sex, and Class",
	)
	fig.update_yaxes(matches=None, rangemode="tozero")
	fig.update_layout(legend_title_text="Sex")
	return fig


def family_groups() -> pd.DataFrame:
	"""Return ticket fare stats grouped by family size and passenger class.

	family_size = SibSp + Parch + 1
	Returns columns: Pclass, family_size, n_passengers, avg_fare, min_fare, max_fare
	Sorted by Pclass then family_size.
	"""

	df = _load_titanic().copy()
	df["family_size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

	grouped = (
		df.groupby(["Pclass", "family_size"], as_index=False)
		.agg(
			n_passengers=("PassengerId", "count"),
			avg_fare=("Fare", "mean"),
			min_fare=("Fare", "min"),
			max_fare=("Fare", "max"),
		)
		.sort_values(["Pclass", "family_size"])  # clear ordering
		.reset_index(drop=True)
	)
	return grouped


def last_names() -> pd.Series:
	"""Return a Series counting occurrences of each passenger last name."""
	df = _load_titanic()
	last = df["Name"].str.split(",", n=1).str[0].str.strip()
	return last.value_counts().sort_values(ascending=False)


def visualize_families():
	"""Plot average fare vs family size across passenger classes.

	Addresses question (written in app):
		How does average ticket fare vary with family size across passenger classes?
	"""

	data = family_groups()
	fig = px.line(
		data,
		x="family_size",
		y="avg_fare",
		color="Pclass",
		markers=True,
		labels={"family_size": "Family Size", "avg_fare": "Average Fare", "Pclass": "Class"},
		title="Average Ticket Fare by Family Size and Class",
	)
	return fig


def determine_age_division() -> pd.DataFrame:
	"""Add a Boolean column 'older_passenger' indicating age above class median.

	Rows with missing Age will have older_passenger set to False (cannot compare).
	Returns the updated DataFrame.
	"""

	df = _load_titanic().copy()
	# Compute class median ages (ignoring NaN)
	class_medians = df.groupby("Pclass")["Age"].median()
	df["older_passenger"] = df.apply(
		lambda r: False if pd.isna(r["Age"]) else r["Age"] > class_medians.loc[r["Pclass"]], axis=1
	)
	return df


def visualize_age_division():
	"""Visualize survival rate by class and relative age division (bonus)."""
	df = determine_age_division()
	grouped = (
		df.groupby(["Pclass", "older_passenger"], as_index=False)
		.agg(n_passengers=("PassengerId", "count"), n_survivors=("Survived", "sum"))
	)
	grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"].replace(0, pd.NA)

	fig = px.bar(
		grouped,
		x="Pclass",
		y="survival_rate",
		color="older_passenger",
		barmode="group",
		labels={"older_passenger": "Older Than Class Median", "survival_rate": "Survival Rate"},
		title="Survival Rate by Class and Age Division",
	)
	fig.update_yaxes(rangemode="tozero")
	return fig


# If executed directly (optional quick smoke test)
if __name__ == "__main__":  # pragma: no cover
	print(survival_demographics().head())
	print(family_groups().head())
	print(last_names().head())