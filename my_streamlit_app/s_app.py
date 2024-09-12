import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pickle

# Streamlit App
def main():
    st.title("User Engagement & Satisfaction Dashboard")

    # Upload the datasets
    engagement_file = st.file_uploader("Upload the engagement dataset (CSV)", type="csv")
    experience_file = st.file_uploader("Upload the experience dataset (CSV)", type="csv")

    if engagement_file and experience_file:
        # Load datasets
        engagement_df = pd.read_csv(engagement_file)
        experience_df = pd.read_csv(experience_file)

        # Display engagement and experience datasets
        st.subheader("Engagement Data")
        st.dataframe(engagement_df.head())

        st.subheader("Experience Data")
        st.dataframe(experience_df.head())

        # Merge datasets on a common identifier
        merged_df = engagement_df.merge(experience_df, left_on='MSISDN/Number', right_on="CustomerID", how='inner')

        st.subheader("Merged Data")
        st.dataframe(merged_df.head())

        # Define columns for KMeans clustering
        engagement_columns = ["SessionDuration", "SessionFrequency", "TotalTraffic"]
        experience_columns = ["AvgTCP", "AvgRTT", "AvgThroughput"]

        # Cluster Engagement Data using KMeans
        k = 3
        X_eng = merged_df[engagement_columns]
        X_exp = merged_df[experience_columns]

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_eng)
        merged_df['cluster'] = cluster_labels

        # Determine less engaged cluster based on average engagement metrics
        merged_df['AVG'] = (merged_df['SessionDuration'] + merged_df['SessionFrequency'] + merged_df['TotalTraffic']) / 3
        grouped = merged_df.groupby('cluster', as_index=False)['AVG'].mean()
        less_engaged_cluster = grouped.loc[grouped['AVG'].idxmin(), 'cluster']

        # Calculate Engagement Score based on distance to less engaged cluster center
        less_engaged_cluster_center = kmeans.cluster_centers_[less_engaged_cluster]
        distances = euclidean_distances(X_eng, [less_engaged_cluster_center]).flatten()
        merged_df['EngagementScore'] = distances

        # Cluster Experience Data using KMeans
        kmeans_exp = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels_exp = kmeans_exp.fit_predict(X_exp)

        # Calculate Experience Score based on distance to worst experience cluster center
        worst_experience_cluster_center = kmeans_exp.cluster_centers_[less_engaged_cluster]
        distances_exp = euclidean_distances(X_exp, [worst_experience_cluster_center]).flatten()
        merged_df['ExperienceScore'] = distances_exp

        # Calculate Satisfaction Score as an average of Engagement and Experience Scores
        merged_df['SatisfactionScore'] = (merged_df['EngagementScore'] + merged_df['ExperienceScore']) / 2

        # Display top 10 users by Satisfaction Score
        st.subheader("Top 10 Users by Satisfaction Score")
        sorted_df = merged_df[['MSISDN/Number', 'SatisfactionScore']].sort_values(by='SatisfactionScore', ascending=False)
        top_10 = sorted_df.head(10)
        st.dataframe(top_10)

        # Plot clusters of Engagement vs Experience Scores
        st.subheader("Engagement vs Experience Clusters")
        plt.figure(figsize=(10, 6))
        cluster_colors = {0: 'red', 1: 'blue'}
        for cluster_label, color in cluster_colors.items():
            cluster_df = merged_df[merged_df["cluster"] == cluster_label]
            plt.scatter(cluster_df["EngagementScore"], cluster_df["ExperienceScore"], c=color, label=f"Cluster {cluster_label}")

        plt.xlabel("Engagement Score")
        plt.ylabel("Experience Score")
        plt.title("2D Scatter Plot of Engagement vs Experience Scores")
        plt.legend()
        st.pyplot(plt)

        # Show average satisfaction and experience per cluster
        st.subheader("Average Satisfaction per Cluster")
        avg_satisfaction = merged_df.groupby('cluster', as_index=False)['SatisfactionScore'].mean()
        st.dataframe(avg_satisfaction.rename(columns={'SatisfactionScore': 'Average Satisfaction'}))

        st.subheader("Average Experience per Cluster")
        avg_experience = merged_df.groupby('cluster', as_index=False)['ExperienceScore'].mean()
        st.dataframe(avg_experience.rename(columns={'ExperienceScore': 'Average Experience'}))

        # Save Satisfaction Score to CSV
        st.subheader("Save Satisfaction Scores to CSV")
        if st.button('Save CSV'):
            merged_df[['MSISDN/Number', 'SatisfactionScore', 'ExperienceScore', 'EngagementScore']].to_csv("satisfaction_score.csv", index=False)
            st.success("File saved as 'satisfaction_score.csv'")

        # Save the KMeans models using pickle
        def save_model(model, filename):
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            st.success(f"Model saved as '{filename}'")

        # Save the Engagement KMeans model
        if st.button('Save Engagement KMeans Model'):
            save_model(kmeans, "engagement_kmeans_model.pkl")

        # Save the Experience KMeans model
        if st.button('Save Experience KMeans Model'):
            save_model(kmeans_exp, "experience_kmeans_model.pkl")

if __name__ == '__main__':
    main()
