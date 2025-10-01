import streamlit as st
import pandas as pd

from apputil import (
	survival_demographics,
	visualize_demographic,
	family_groups,
	last_names,
	visualize_families,
	visualize_age_division,
)

st.set_page_config(page_title="Titanic Analysis", layout="wide")

st.title("Titanic Analysis – Week 5")

# Section 1: Survival Demographics
st.header("Survival Demographics")
st.write(
	"Question: How does survival rate vary across passenger class and sex for each age group?"
)
demo_tbl = survival_demographics()
st.dataframe(demo_tbl, use_container_width=True)
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

# Section 2: Family Size and Wealth
st.header("Family Size and Wealth")
st.write(
	"Question: How does average ticket fare vary with family size across passenger classes?"
)
fam_tbl = family_groups()
st.dataframe(fam_tbl, use_container_width=True)

# Last name frequency insight
st.subheader("Most Common Last Names")
ln_series = last_names()
st.write(ln_series.head(15))
st.caption(
	"Comparing last name frequency with the family size table helps identify large families or groups traveling together."
)
# Brief findings (dynamic summary)
if not ln_series.empty:
	most_common = ln_series.index[0]
	most_common_count = int(ln_series.iloc[0])
	max_family_size = int(fam_tbl["family_size"].max())
	st.write(
		f"The most common last name is {most_common} (appears {most_common_count} times). "
		f"The largest single family_size in the dataset is {max_family_size}. "
		"Multiple occurrences of a surname do not always imply one large family; they can represent smaller nuclear families or unrelated passengers sharing a last name."
	)
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)

# Bonus: Age division relative to class median
st.header("Bonus: Age Division Relative to Class Median")
st.write(
	"Question: Within each class, do passengers older than the class median age have different survival rates?"
)
fig3 = visualize_age_division()
st.plotly_chart(fig3, use_container_width=True)