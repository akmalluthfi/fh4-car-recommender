import streamlit as st
from utils import load_data, preprocessing, get_recommendations, display_car_card

st.set_page_config(
    page_title="FH4 Car Recommender",
    layout="wide",
    page_icon="https://cdn2.steamgriddb.com/icon/7463afe23eae7efe3c72737a5d3d693f.ico",
)


st.title("Forza Horizon 4 Car Recommender")
st.text(
    "This application recommends similar cars based on your selected vehicle in Forza Horizon 4 using a content-based filtering approach."
)

df = load_data()
similarity_matrix = preprocessing(df)

selected_car = st.selectbox("Select your car:", df["name"].sort_values().unique())

if selected_car:
    car = df[df["name"] == selected_car].iloc[0]
    display_car_card(car)

if st.button("Show Recommendation"):
    st.text("Here are few Recommendations..")
    recommendations = get_recommendations(similarity_matrix, car)

    for _, row in df.iloc[recommendations].iterrows():
        display_car_card(row)
