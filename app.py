import streamlit as st
import numpy as np
from spotify_client.client import create_spotify_client
from recommendation.recommendation import get_top_tracks, get_audio_features, collaborative_filtering_recommendation, content_based_recommendation
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.title("Spotify Music Recommendation System")

# Create Spotify client
sp = create_spotify_client()

if 'user_factors' not in st.session_state:
    st.session_state.user_factors = None
if 'item_factors' not in st.session_state:
    st.session_state.item_factors = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None

def load_data():
    with st.spinner('Loading data...'):
        track_ids = get_top_tracks(sp)
        audio_features = get_audio_features(sp, track_ids)
        return track_ids, audio_features

def train_models(audio_features):
    svd = TruncatedSVD(n_components=20)
    user_factors = svd.fit_transform(audio_features)
    item_factors = svd.components_
    similarity_matrix = cosine_similarity(audio_features)
    return user_factors, item_factors, similarity_matrix

track_ids, audio_features = load_data()

if st.button("Train Models"):
    st.session_state.user_factors, st.session_state.item_factors, st.session_state.similarity_matrix = train_models(audio_features)
    st.success("Models trained successfully!")

if st.session_state.user_factors is not None and st.session_state.item_factors is not None:
    user_id = st.number_input("Enter User ID", min_value=0, max_value=len(st.session_state.user_factors)-1, step=1, value=0)
    if st.button("Get Recommendations"):
        recommendations = collaborative_filtering_recommendation(st.session_state.user_factors, st.session_state.item_factors, user_id)
        st.write("Collaborative Filtering Recommendations:")
        for idx in recommendations:
            st.write(track_ids[idx])

    track_id = st.selectbox("Select Track ID", options=list(range(len(track_ids))))
    if st.button("Get Similar Tracks"):
        similar_tracks = content_based_recommendation(st.session_state.similarity_matrix, track_id)
        st.write("Content-Based Recommendations:")
        for idx in similar_tracks:
            st.write(track_ids[idx])
