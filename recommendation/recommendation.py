import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def get_top_tracks(sp, limit=50):
    top_tracks = sp.current_user_top_tracks(limit=limit)
    track_ids = [track['id'] for track in top_tracks['items']]
    return track_ids

def get_audio_features(sp, track_ids):
    audio_features = sp.audio_features(track_ids)
    df = pd.DataFrame(audio_features)
    df = (df - df.mean()) / df.std()
    return df

def collaborative_filtering_recommendation(user_factors, item_factors, user_id, top_n=10):
    user_vector = user_factors[user_id]
    scores = np.dot(user_vector, item_factors.T)
    top_items = np.argsort(scores)[-top_n:]
    return top_items

def content_based_recommendation(similarity_matrix, track_id, top_n=10):
    similar_tracks = np.argsort(similarity_matrix[track_id])[-top_n:]
    return similar_tracks
