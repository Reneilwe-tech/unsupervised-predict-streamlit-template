"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('../unsupervised_data/unsupervised_movie_data/movies.csv)
df_movies = pd.read_csv('resources/data/movies.csv')
ratings = pd.read_csv('resources/data/ratings.csv')
#df_train = pd.read_csv('../unsupervised_data/unsupervised_movie_data/train.csv') 

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Recommender System","Data Visualization","Genres"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":          
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # if page_selection == "Solution Overview":
    #     st.title("Solution Overview")
    #     st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
     #Data visualisation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if page_selection == "Data Visualization":
        st.title('Data Summary')
        st.image('resources/imgs/giphy.gif',use_column_width=True)
        st.write('## Graphs')

        # Title word cloud graph
        st.write('Title Word Cloud')

        # df_movies['title'] = df_movies['title'].astype('str')
        # title_corpus = ' '.join(df_movies['title'])
        # title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
        # plt.figure(figsize=(16,8))
        # plt.imshow(title_wordcloud)
        # plt.axis('off')
        # st.pyplot()

        # place a picture intead of drawing the grasph as it takes too long
        st.image('resources/imgs/word_cloud_unsupervised.png',use_column_width=True)
        st.markdown("""The word cloud shows most frequently occuring words in movie titles. The title of a movie as we know is
                    the first thing that grabs a customer's attention hence some movie titles with certain words are considered 
                    more potent than the others. From this word cloud it can be observed that the most commonly used word is Love followed by Girl, and Man.""")

        # Number of movies per year graph
        # year_count = df_movies.groupby('year')['title'].count()
        # plt.figure(figsize=(18,5))
        # year_count.plot()


        # Most common to least common genre graph
        # Create dataframe containing only the movieId and genres
        movies_genres = pd.DataFrame(df_movies[['movieId', 'genres']],columns=['movieId', 'genres'])
        movies_genres = pd.DataFrame([(tup.movieId, d) for tup in movies_genres.itertuples() for d in tup.genres],
                             columns=['movieId', 'genres'])
        # Plot the genres from most common to least common
        plot = plt.figure(figsize=(15, 10))
        plt.title('Most Common Genres\n', fontsize=20)
        sns.countplot(y="genres", data=movies_genres,order=movies_genres['genres'].value_counts(ascending=False).index,palette='Reds_r')
        st.pyplot()
        st.markdown("""The graph shows the most common genres in movies. The most common genre is Drama and the least common is Imax""")

        # Ratings distribution graph
        #ratings = pd.DataFrame(df_train[['rating']],columns=['rating'])
        plt.figure(figsize=(15, 10))
        plt.title('Ratings Distribution \n', fontsize=20)
        sns.countplot(x="rating", data=ratings, palette='Reds_r')
        st.pyplot()
        st.markdown("""This graph shows the distribution of ratings in the data.
                    It can be seen that more movies are rated 4.0 and 3.0 while fewer movies are rated 0.5 and 1.5 """)

        #Ratings per movie graph
        #Because there are so many movies in the database, we clip it at 100, otherwise the graph becomes too difficult to read
        # ratings_per_movie = pd.DataFrame(df_train.groupby('movieId')['rating'].count().clip(upper=100))
        # ratings_per_movie.groupby('rating')['rating'].count()
        # plt.figure(figsize=(15, 10))
        # plt.title('Ratings per Movie \n', fontsize=20)
        # sns.histplot(data=ratings_per_movie, palette='Reds_r')
        # st.pyplot()

    # get top 10 rated movies per genre
    if  page_selection == "Genres":
        st.title('Top 10 Movies Per Genre')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
  
        genres = ['Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','Foreign','History','Horror','Music','Mystery','Romance','Science Fiction','TV Movie','Thriller','War']
        genre_selection = st.selectbox('Select Genre', genres)  
        # join ratings and movies dataframe
        new_df = pd.merge(df_movies,ratings,on='movieId',how='inner')
        
        if  genre_selection:
            genre_selection_list = [genre_selection]
            result= new_df[new_df['genres'].isin(genre_selection_list)]

            # sort the result dataframe
            result = result.sort_values('rating', ascending=False )         
            result = result.drop(['userId','timestamp'],axis=1)  # drop the userId and times stamp columns    
            st.dataframe(result.head(10))


    if page_selection == "Home":
        st.title('About')
        st.image('resources/imgs/giphy.gif',use_column_width=True)
        st.markdown("""Do you feel like watching a movie but unsure what to watch? or have a specific genre in mind? Well, this particular app is designed for you!!!
                     It uses intelligent algorithms to accurately recommend the best movies to you based on what you like to watch.""")


if __name__ == '__main__':
    main()
