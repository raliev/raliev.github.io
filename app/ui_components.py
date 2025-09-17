# app/ui_components.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def render_sidebar():
    """
    Renders the sidebar with all user controls and returns the selections.
    """
    st.sidebar.header("Controls")

    # --- Data Source Selection ---
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Demo Matrix", "Load MovieLens CSV"],
        help="Choose between a small, editable matrix or the full MovieLens dataset."
    )

    # --- Algorithm Selection ---
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        ("SVD", "ALS", "ALS (Improved)", "ALS (PySpark)", "BPR", "ItemKNN", "UserKNN", "Slope One", "NMF",
         "FunkSVD", "PureSVD", "SVD++", "WRMF", "CML",
         "NCF / NeuMF", "SASRec", "VAE", "SLIM") # --- ADDITIONS ---
    )

    # Initialize dictionaries for parameters
    model_params = {}
    data_params = {}
    data = None

    if data_source == "Demo Matrix":
        initial_data = {
            'Movie 1': [5, 4, 0, 1], 'Movie 2': [4, 5, 1, 0],
            'Movie 3': [0, 0, 5, 4], 'Movie 4': [1, 0, 4, 5]
        }
        users = ['User 1', 'User 2', 'User 3', 'User 4']
        data = pd.DataFrame(initial_data, index=users)
        st.sidebar.subheader("1. Edit User Data")
        data = st.sidebar.data_editor(data, num_rows="dynamic")
    else:
        st.sidebar.info("The MovieLens dataset will be downloaded automatically on the first run.")
        st.sidebar.subheader("1. Select Data Size")
        data_params['num_users'] = st.sidebar.slider("Number of Users to Use", 50, 610, 100)
        data_params['num_movies'] = st.sidebar.slider("Number of Movies to Use", 100, 2000, 500)

    # --- Hyperparameters ---
    st.sidebar.subheader(f"2. {algorithm} Hyperparameters")

    if algorithm in ["SVD", "ALS", "ALS (Improved)", "BPR", "NMF", "ALS (PySpark)", "FunkSVD", "PureSVD", "SVD++", "WRMF", "CML", "NCF / NeuMF", "SASRec", "VAE"]:
        model_params['k'] = st.sidebar.slider("Latent Factors (k)", 1, 100, 32)

    if algorithm == "ALS":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 30, 10)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 1.0, 0.1, 0.01)
    elif algorithm == "ALS (Improved)":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 30, 10)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (Factors)", 0.0, 1.0, 0.05, 0.01)
        model_params['lambda_biases'] = st.sidebar.slider("Regularization (Biases)", 0.0, 20.0, 10.0, 0.5)
    elif algorithm == "ALS (PySpark)":
        model_params['iterations'] = st.sidebar.slider("Iterations (maxIter)", 5, 25, 10)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (regParam)", 0.01, 0.2, 0.1, 0.01)
    elif algorithm == "BPR":
        model_params['iterations'] = st.sidebar.slider("Iterations", 100, 5000, 1000, 100)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate (alpha)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 0.1, 0.01, 0.001, format="%.3f")
    elif algorithm in ["ItemKNN", "UserKNN"]:
        model_params['k'] = st.sidebar.slider("Number of Neighbors (k)", 5, 100, 20)
        model_params['similarity_metric'] = st.sidebar.selectbox(
            "Similarity Metric",
            ("cosine", "adjusted_cosine", "pearson")
        )
        if algorithm == "ItemKNN":
            model_params['min_support'] = st.sidebar.slider("Minimum Support", 0, 10, 2)
            model_params['shrinkage'] = st.sidebar.slider("Shrinkage", 0.0, 100.0, 0.0, 1.0)
    elif algorithm == "NMF":
        model_params['max_iter'] = st.sidebar.slider("Max Iterations", 50, 500, 200)
    elif algorithm == "FunkSVD":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 30, 10)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.005, 0.001)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 1.0, 0.02, 0.01)
    elif algorithm == "SVD++":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 30, 20)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.005, 0.001)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 1.0, 0.02, 0.01)
    elif algorithm == "WRMF":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 30, 10)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 1.0, 0.1, 0.01)
        model_params['alpha'] = st.sidebar.slider("Alpha", 1, 100, 40)
    elif algorithm == "CML":
        model_params['iterations'] = st.sidebar.slider("Iterations", 1, 200, 100, 10)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        model_params['lambda_reg'] = st.sidebar.slider("Regularization (lambda)", 0.0, 1.0, 0.01, 0.001)
        model_params['margin'] = st.sidebar.slider("Margin", 0.1, 2.0, 0.5, 0.1)
    elif algorithm == "NCF / NeuMF":
        model_params['model_type'] = st.sidebar.selectbox("Model Type", ('NeuMF', 'GMF', 'NCF'))
        model_params['epochs'] = st.sidebar.slider("Epochs", 1, 50, 10)
        model_params['batch_size'] = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=64)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    elif algorithm == "SASRec":
        model_params['epochs'] = st.sidebar.slider("Epochs", 5, 100, 30)
        model_params['batch_size'] = st.sidebar.select_slider("Batch Size", options=[32, 64, 128, 256], value=128)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        model_params['max_len'] = st.sidebar.slider("Max Sequence Length", 10, 200, 50)
        model_params['num_blocks'] = st.sidebar.slider("Number of Attention Blocks", 1, 4, 2)
        model_params['num_heads'] = st.sidebar.slider("Number of Attention Heads", 1, 4, 1)
        model_params['dropout_rate'] = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    elif algorithm == "SLIM":
        st.info("SLIM learns a sparse item-item similarity matrix. L1 encourages sparsity, L2 prevents large weights.")
        model_params['l1_reg'] = st.sidebar.slider("L1 Regularization", 0.0, 0.1, 0.001, format="%.4f")
        model_params['l2_reg'] = st.sidebar.slider("L2 Regularization", 0.0, 0.1, 0.0001, format="%.5f")
    elif algorithm == "VAE":
        model_params['epochs'] = st.sidebar.slider("Epochs", 1, 100, 20)
        model_params['batch_size'] = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=64)
        model_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")


    run_button = st.sidebar.button("Run Algorithm", use_container_width=True)

    return data_source, data, algorithm, model_params, data_params, run_button


def get_movie_title(movie_id, movie_titles_df):
    """Helper function to get movie title from movieId."""
    if movie_titles_df is None:
        return f"Movie ID: {movie_id}"
    try:
        return movie_titles_df.loc[movie_id, 'title']
    except KeyError:
        return f"Movie ID: {movie_id} (Unknown)"


def render_results_tabs(results):
    st.header("Results")
    tab1, tab2, tab3 = st.tabs(["Model Internals", "Predictions", "Recommendations"])

    movie_titles_df = results.get('movie_titles')

    def rename_columns_to_titles(df):
        if movie_titles_df is not None:
            title_map = {mid: get_movie_title(mid, movie_titles_df) for mid in df.columns}
            return df.rename(columns=title_map)
        return df

    with tab1:
        st.subheader(f"Inside the {results['algo_name']} Model")

        # 1. For Factorization Models
        if results['algo_name'] in ["SVD", "PureSVD", "ALS", "ALS (Improved)", "NMF", "FunkSVD", "SVD++", "WRMF", "BPR", "CML"]:
            st.write("These models decompose the original matrix into User-Factors (P) and Item-Factors (Q).")
            if results.get('P') is not None and results.get('Q') is not None:
                q_df = pd.DataFrame(results['Q'], index=results['original_df'].columns)
                q_df_renamed = rename_columns_to_titles(q_df.T).T
                col1, col2 = st.columns(2)
                with col1: st.write("**P (Users x Factors)**"); st.dataframe(pd.DataFrame(results['P'], index=results['original_df'].index).style.format("{:.2f}"))
                with col2: st.write("**Q (Items x Factors)**"); st.dataframe(q_df_renamed.style.format("{:.2f}"))
            else:
                st.info(f"{results['algo_name']} does not expose user/item factor matrices in a simple format.")

        # 2. For Neighborhood / Similarity Models
        elif results['algo_name'] in ["ItemKNN", "UserKNN", "SLIM"]:
            st.write("These models compute or learn a **Similarity Matrix** to find similar users or items.")
            sim_matrix = results.get('similarity_matrix')
            if sim_matrix is not None:
                df = pd.DataFrame(sim_matrix)
                if results['algo_name'] in ["ItemKNN", "SLIM"]:
                    df.index = results['original_df'].columns
                    df.columns = results['original_df'].columns
                    df = rename_columns_to_titles(rename_columns_to_titles(df.T).T)
                else:
                    df.index = results['original_df'].index
                    df.columns = results['original_df'].index

                st.write(f"**Learned Similarity Matrix (subset)**")
                max_dim = 25
                if df.shape[0] > max_dim:
                    st.info(f"Displaying a {max_dim}x{max_dim} subset of the full similarity matrix.")
                    df_subset = df.iloc[:max_dim, :max_dim]
                else:
                    df_subset = df

                fig = px.imshow(df_subset, text_auto=".2f", aspect="auto", title=f"{results['algo_name']} Similarity")
                st.plotly_chart(fig)
            else:
                st.info("Similarity matrix not available.")

        # 3. For Autoencoder Models
        elif results['algo_name'] in ["VAE"]:
            st.write("Autoencoder models learn to **reconstruct** a user's interaction history from a compressed latent representation.")
            recon_matrix = results.get('reconstructed_matrix')
            if recon_matrix is not None:
                user_list = results['original_df'].index
                selected_user = st.selectbox("Select a User to Visualize:", options=user_list)

                original_vec = results['original_df'].loc[selected_user]
                recon_vec = pd.Series(recon_matrix[user_list.get_loc(selected_user)], index=original_vec.index)

                rated_items = original_vec[original_vec > 0].index
                top_unrated_recon = recon_vec.drop(rated_items).nlargest(10).index
                items_to_show = rated_items.union(top_unrated_recon)

                vis_df = pd.DataFrame({
                    'Original Interaction': (original_vec[items_to_show] > 0).astype(int),
                    'Reconstructed Score': recon_vec[items_to_show]
                })
                vis_df.index = vis_df.index.map(lambda mid: get_movie_title(mid, movie_titles_df))

                fig = px.bar(vis_df, barmode='group', title=f"Original vs. Reconstructed Interactions for User {selected_user}")
                st.plotly_chart(fig)
            else:
                st.info("Reconstructed matrix not available.")

        else:
            st.info(f"No specific internal visualization is available for {results['algo_name']}.")

    with tab2:
        st.subheader("Original Data vs. Predicted Scores")
        if results['algo_name'] == 'BPR': st.warning("Reminder: BPR outputs scores for ranking, not predicted ratings.")

        original_df, predicted_df = results['original_df'], results['predicted_df']
        max_users, max_items = 20, 20

        original_df_renamed = rename_columns_to_titles(original_df)
        predicted_df_renamed = rename_columns_to_titles(predicted_df)

        if original_df.shape[0] > max_users or original_df.shape[1] > max_items:
            st.info(f"Displaying a subset ({max_users} users, {max_items} items) of the full matrix for performance.")
            display_original_df = original_df_renamed.iloc[:max_users, :max_items]
            display_predicted_df = predicted_df_renamed.iloc[:max_users, :max_items]
        else:
            display_original_df, display_predicted_df = original_df_renamed, predicted_df_renamed

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data (subset)**")
            st.dataframe(display_original_df)
        with col2:
            st.write("**Predictions (subset, new items highlighted)**")
            def style_predictions(row):
                original_row = display_original_df.loc[row.name]
                highlight = 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;'
                return [highlight if original_row[col] == 0 else '' for col in original_row.index]
            st.dataframe(display_predicted_df.style.format("{:.2f}").apply(style_predictions, axis=1))

    with tab3:
        st.subheader("Get Top N Recommendations")
        user_list = results['predicted_df'].index
        selected_user = st.selectbox("Select a User:", options=user_list)
        num_recs = st.number_input("Number of Recommendations (N):", 1, 20, 5)

        if selected_user:
            original_data, user_scores = results['original_df'], results['predicted_df'].loc[selected_user]
            seen_items = original_data.loc[selected_user][original_data.loc[selected_user] > 0].index
            top_n_ids = user_scores.drop(seen_items, errors='ignore').nlargest(num_recs)

            top_n_titles = top_n_ids.copy()
            top_n_titles.index = top_n_titles.index.map(lambda mid: get_movie_title(mid, movie_titles_df))

            st.write(f"**Top {num_recs} recommendations for User {selected_user}:**")
            st.dataframe(top_n_titles.to_frame(name="Predicted Score").style.format("{:.2f}"))


def render_performance_tab(metrics):
    st.header("Model Performance on Test Set")
    if metrics['type'] == 'explicit':
        st.info("These metrics evaluate the accuracy of the predicted ratings against the actual ratings in the test set.")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics.get('rmse', 0):.4f}", help="Measures the average error in predicted ratings. Lower is better.")
        col2.metric(label="Mean Absolute Error (MAE)", value=f"{metrics.get('mae', 0):.4f}", help="Similar to RMSE, but less sensitive to large errors. Lower is better.")
        col3.metric(label="R-squared (R²)", value=f"{metrics.get('r2', 0):.4f}", help="Indicates the proportion of variance in the actual ratings that is predictable from the model. Closer to 1 is better.")

        col4, col5 = st.columns(2)
        col4.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{metrics.get('mape', 0):.2f}%", help="Expresses the mean absolute error as a percentage of actual values. Lower is better.")
        col5.metric(label="Explained Variance Score", value=f"{metrics.get('explained_variance', 0):.4f}", help="Measures how well the model accounts for the variation in the original data. Closer to 1 is better.")

    elif metrics['type'] == 'implicit':
        st.info("These metrics evaluate the quality of the item rankings produced by the model.")
        col1, col2 = st.columns(2)
        k_val = metrics['k']
        col1.metric(label=f"Precision@{k_val}", value=f"{metrics['precision']:.2%}")
        col2.metric(label=f"Recall@{k_val}", value=f"{metrics['recall']:.2%}")
        st.info(f"**Precision**: Of the top {k_val} items recommended, what percentage were actually relevant items from the test set?\n\n**Recall**: Of all the relevant items in the test set, what percentage did the model successfully recommend in the top {k_val}?")
        fig = go.Figure(data=[go.Bar(name='Precision', x=['Performance'], y=[metrics['precision']]), go.Bar(name='Recall', x=['Performance'], y=[metrics['recall']])])
        fig.update_layout(title_text=f'Precision and Recall @ {k_val}', yaxis_title="Score", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)