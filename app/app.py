# app.py
import streamlit as st
import pandas as pd
import numpy as np
from ui_components import render_sidebar, render_results_tabs, render_performance_tab
from utils import download_and_load_movielens, split_data, calculate_regression_metrics, precision_recall_at_k
from algorithms import (
    SVDRecommender, ALSRecommender, BPRRecommender, ALSImprovedRecommender,
    ItemKNNRecommender, SlopeOneRecommender, NMFRecommender, ALSPySparkRecommender,
    FunkSVDRecommender, PureSVDRecommender, SVDppRecommender, WRMFRecommender, CMLRecommender,
    UserKNNRecommender, NCFRecommender, SASRecRecommender, SLIMRecommender, VAERecommender
)

# --- Page Configuration ---
st.set_page_config(page_title="Recommender System Lab", layout="wide")
st.title("Recommender System Laboratory")

# --- Initialize Session State ---
if 'results' not in st.session_state: st.session_state['results'] = None
if 'metrics' not in st.session_state: st.session_state['metrics'] = None
if 'movie_titles' not in st.session_state: st.session_state['movie_titles'] = None


# --- Render Sidebar and Get User Inputs ---
data_source, data, algorithm, model_params, data_params, run_button = render_sidebar()

# --- Main Logic ---
if run_button:
    progress_bar_placeholder = st.empty()

    with st.spinner("Preparing data..."):
        if data_source == "Load MovieLens CSV":
            full_df, movie_titles_df = download_and_load_movielens()
            st.session_state['movie_titles'] = movie_titles_df
            if full_df is not None:
                num_users = data_params.get('num_users', 100)
                num_movies = data_params.get('num_movies', 500)

                movie_counts = full_df[full_df > 0].count(axis=0)
                top_movies_ids = movie_counts.nlargest(num_movies).index
                df_filtered_movies = full_df[top_movies_ids]

                user_counts = df_filtered_movies[df_filtered_movies > 0].count(axis=1)
                top_users_ids = user_counts.nlargest(num_users).index
                data_to_use = df_filtered_movies.loc[top_users_ids]

                st.write(f"Using a subset of the data: **{data_to_use.shape[0]} users** and **{data_to_use.shape[1]} movies**.")
                train_df, test_df = split_data(data_to_use)
                data_to_train = train_df.to_numpy()
        else:
            st.session_state['movie_titles'] = None
            train_df, test_df, data_to_use = data, None, data
            data_to_train = train_df.to_numpy()

    if 'data_to_use' in locals() and data_to_use is not None:
        progress_bar = progress_bar_placeholder.progress(0, text=f"Training {algorithm} model...")

        model_map = {
            "SVD": SVDRecommender, "ALS": ALSRecommender, "ALS (Improved)": ALSImprovedRecommender,
            "BPR": BPRRecommender, "ItemKNN": ItemKNNRecommender, "Slope One": SlopeOneRecommender,
            "NMF": NMFRecommender, "ALS (PySpark)": ALSPySparkRecommender,
            "FunkSVD": FunkSVDRecommender, "PureSVD": PureSVDRecommender, "SVD++": SVDppRecommender,
            "WRMF": WRMFRecommender, "CML": CMLRecommender,
            "UserKNN": UserKNNRecommender,
            "NCF / NeuMF": NCFRecommender,
            "SASRec": SASRecRecommender,
            "SLIM": SLIMRecommender,
            "VAE": VAERecommender
        }

        model_class = model_map.get(algorithm)
        if model_class:
            if 'k' not in model_params: model_params['k'] = 0
            model = model_class(**model_params)
        else:
            st.error(f"Algorithm {algorithm} not found."); st.stop()

        try:
            model.train_data = data_to_train
            model.fit(data_to_train, progress_callback=lambda p: progress_bar.progress(p, text=f"Training {algorithm} model... {int(p*100)}%"))
        except ImportError as e:
            st.error(f"Could not run {algorithm}. Please make sure required libraries are installed: {e}")
            st.stop()

        progress_bar.empty()
        predicted_matrix = model.predict()
        predicted_matrix = np.nan_to_num(predicted_matrix)
        predicted_df = pd.DataFrame(predicted_matrix, index=data_to_use.index, columns=data_to_use.columns)

        st.session_state['results'] = {
            'algo_name': model.name, 'predicted_df': predicted_df, 'original_df': data_to_use,
            'P': getattr(model, 'P', None), 'Q': getattr(model, 'Q', None), 'sigma': getattr(model, 'sigma', None),
            'similarity_matrix': getattr(model, 'similarity_matrix', None),
            'reconstructed_matrix': getattr(model, 'reconstructed_matrix', None),
        }

        if test_df is not None:
            if algorithm in ["BPR", "CML", "WRMF", "NCF / NeuMF", "SASRec", "VAE", "SLIM"]:
                k_prec_rec = 10
                precision, recall = precision_recall_at_k(predicted_df, test_df, k=k_prec_rec)
                st.session_state['metrics'] = {'type': 'implicit', 'precision': precision, 'recall': recall, 'k': k_prec_rec}
            else:
                metrics = calculate_regression_metrics(predicted_df, test_df)
                st.session_state['metrics'] = {'type': 'explicit', **metrics}
        else:
            st.session_state['metrics'] = None


# --- Display Results ---
if st.session_state['results']:
    results_with_titles = {
        **st.session_state['results'],
        'movie_titles': st.session_state.get('movie_titles')
    }
    if st.session_state['metrics']:
        main_tabs = st.tabs(["Results", "Performance"])
        with main_tabs[0]:
            render_results_tabs(results_with_titles)
        with main_tabs[1]:
            render_performance_tab(st.session_state['metrics'])
    else:
        render_results_tabs(results_with_titles)
else:
    st.info("Select your data, algorithm, and parameters in the sidebar, then click 'Run'.")