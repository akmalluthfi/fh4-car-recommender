import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@st.cache_data
def load_data():
    return pd.read_csv("data/fh4_cars.csv")


@st.cache_data
def preprocessing(df):
    # Encoding
    encoded_df = pd.get_dummies(df[["category", "transmission"]])
    class_order = [["D", "C", "B", "A", "S1", "S2"]]
    encoded_df["class"] = OrdinalEncoder(categories=class_order).fit_transform(
        df[["class"]]
    )

    # Scaling data
    df_num_col = df.select_dtypes(include=["int64", "float64"]).columns
    scaled_df = pd.DataFrame(
        StandardScaler().fit_transform(df[df_num_col]), columns=df_num_col
    )

    combined_df = pd.concat([scaled_df, encoded_df], axis=1)
    return cosine_similarity(combined_df)


def get_recommendations(similarity_matrix, selected, n=5):
    # Get the pairwise similarity scores of all movies with the given movie
    sim_scores = list(enumerate(similarity_matrix[selected.name]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the n most similar movies
    sim_scores = sim_scores[1 : n + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top n most similar movies
    return movie_indices


def display_car_card(car):
    with st.columns(1, border=True)[0]:
        col1, col2, col3, col4 = st.columns([1, 4, 2, 4])
        with col1:
            st.image(car["images"], use_container_width=True)

        with col2:
            st.markdown(f"##### **{car['name']}**  \n")
            st.markdown(
                f"{car['category']}  \n" f"**Class:** {car['class']} | {car['pi']}"
            )

        with col3:
            st.markdown(
                f"**Power:** {car['power_hp']:,} HP  \n"
                f"**Weight:** {car['weight_lbs']:,} lbs  \n"
                f"**Transmissions:** {car['transmission']}"
            )

        with col4:
            st.markdown(f"##### {car['price']:,} Cr")
            st.markdown(
                f"**Speed:** :violet[{car['speed']}] &nbsp;&nbsp; "
                f"**Handling:** :orange[{car['handling']}] &nbsp;&nbsp; "
                f"**Accel:** :green[{car['acceleration']}] &nbsp;&nbsp; "
                f"**Braking:** :blue"
            )
