import pandas as pd
from math import sqrt
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.spatial import distance


class ExplainableRecommender(object):
    def __init__(self):
        self.preprocessing()
        self.content_recommender()

    def preprocessing(self):
        # Storing the movie information into a pandas dataframe
        movies_df = pd.read_csv('ml-latest/movies.csv')
        # Using regular expressions to find a year stored between parentheses
        # We specify the parantheses so we don't conflict with movies that have years in their titles
        movies_df['year'] = movies_df.title.str.extract(
            '(\(\d\d\d\d\))', expand=False)
        # Removing the parentheses
        movies_df['year'] = movies_df.year.str.extract(
            '(\d\d\d\d)', expand=False)
        # Removing the years from the 'title' column
        movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
        # Applying the strip function to get rid of any ending whitespace characters that may have appeared
        movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

        # Every genre is separated by a | so we simply have to call the split function on |
        movies_df['genres'] = movies_df.genres.str.split('|')

        # Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
        moviesWithGenres_df = movies_df.copy()

        # For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
        for index, row in movies_df.iterrows():
            for genre in row['genres']:
                moviesWithGenres_df.at[index, genre] = 1

        self.movies_df = movies_df
        # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
        self.moviesWithGenres_df = moviesWithGenres_df.fillna(0)

        # Storing the user information into a pandas dataframe
        ratings_df = pd.read_csv('ml-latest/ratings.csv')
        ratings_df = ratings_df.drop('timestamp', 1)
        self.ratings_df = ratings_df

    def content_recommender(self):
        inputMovies = self.get_interests()
        # print(temp_inputMovies)
        inputMovies = inputMovies.reset_index(drop=True)
        # Filtering out the movies by title
        inputId = self.movies_df[self.movies_df['title'].isin(
            inputMovies['title'].tolist())]
        # Then merging it so we can get the movieId. It's implicitly merging it by title.
        inputMovies = pd.merge(inputId, inputMovies)
        # print(inputMovies)

        # Dropping information we won't use from the input dataframe
        inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

        # Final input dataframe
        # If a movie you added in above isn't here, then it might not be in the original
        # dataframe or it might spelled differently, please check capitalisation.
        # Filtering out the movies from the input
        userMovies = self.moviesWithGenres_df[self.moviesWithGenres_df['movieId'].isin(
            inputMovies['movieId'].tolist())]
        # Resetting the index to avoid future issues

        # inspection
        movieId = userMovies['movieId'].tolist()
        title = userMovies['title'].tolist()
        genres = userMovies['genres'].tolist()
        ratings = inputMovies['rating'].tolist()

        # dictionary of lists
        dict = {'movieId': movieId, 'title': title,
                'genres': genres, 'rating': ratings}

        inspect_userMovies = pd.DataFrame(dict)
        inspect_userMovies = inspect_userMovies.sort_values(
            by='rating', ascending=False)
        print(inspect_userMovies.to_string())

        print('Creating Genre Table...')
        userMovies = userMovies.reset_index(drop=True)
        # Dropping unnecessary issues due to save memory and to avoid issues
        userGenreTable = userMovies.drop('movieId', 1).drop(
            'title', 1).drop('genres', 1).drop('year', 1)

        print('Finding highest rated tag...')
        # Dot produt to get weights
        userProfile = userGenreTable.transpose().dot(inputMovies['rating'])


        # The user profile
        highest_rated_tag = userProfile.sort_values(ascending=False)
    
        print("Weighted Genres List")
        print("==========================================")
        print(highest_rated_tag)
        print("------------------------------------------")

        highest_rated_genres=list(highest_rated_tag.index)

        self.reason = " because You like "+str(highest_rated_genres[0])+" most also "+ str(highest_rated_genres[1])+" and "+str(highest_rated_genres[2])
        
        self.userProfile = userProfile

        print('Generating recommendation...')
        self.generate_recommendation()

    def generate_recommendation(self):
        # Now let's get the genres of every movie in our original dataframe
        genreTable = self.moviesWithGenres_df.set_index(
            self.moviesWithGenres_df['movieId'])
        # And drop the unnecessary information
        genreTable = genreTable.drop('movieId', 1).drop(
            'title', 1).drop('genres', 1).drop('year', 1)

        # Multiply the genres by the weights and then take the weighted average
        recommendationTable_df = (
            (genreTable*self.userProfile).sum(axis=1))/(self.userProfile.sum())

        # Sort our recommendations in descending order
        recommendationTable_df = recommendationTable_df.sort_values(
            ascending=False)

        recommendation_no = 1
        # The final recommendation table
        final_recommendation = self.movies_df.loc[self.movies_df['movieId'].isin(
            recommendationTable_df.head(recommendation_no).keys())]

        title = final_recommendation['title'].tolist()
        genres = final_recommendation['genres'].tolist()
        dict = {'title': title, 'genres': genres}

        inspect_final_recommendation = pd.DataFrame(dict)

        for i in range(recommendation_no):
        	print(title[i])
        	print(genres[i])

        	print("---------------------------------------------------------------------------------------------")

        for i in range(recommendation_no):
            recommendation = final_recommendation['title'].values[i]
            full_statement = "we recommend " + recommendation+self.reason
            print(full_statement)
        print("---------------------------------------------------------------------------------------------")
        print("-----------------------------------------END--------------------------------------------------")
        print("----------------------------------------------------------------------------------------------")



    def build_userprofile(self, movie_interests, ratings):
        userInput = []
        for idx in range(len(movie_interests)):
            movie_title = movie_interests[idx]
            rating = float(ratings[idx])
            userInput.append({'title': movie_title, 'rating': rating})
        inputMovies = pd.DataFrame(userInput)
        return inputMovies

    def get_interests(self):
        movie_interests = []
        ratings = []

        print("----------------------------------------------------------------------------------------------------")
        print("-------------------------PLEASE ENTER MOVIES IN CAPITAL LETTERS-------------------------------------")
        print("--------IF Starts with a \"The\", like \"The Matrix\", write like this \"Matrix\, The\"-------------")
        print("-----------------------------------------------------------------------------------------------------")
        print("------------------------------------------MENU------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------------")

        n = input("Enter number of ratings you want to give: ")
        n = int(n)
        print("Please enter movies and rating")
        for i in range(n):
            movie = input("Enter movie interest"+str(i)+": ")
            rating = input("Rate 1 to 5: ")
            movie_interests.append(movie)
            ratings.append(rating)

        input_movies = self.build_userprofile(movie_interests, ratings)
        print(input_movies)
        print("-----------------------------------")
        return input_movies


if __name__ == "__main__":
    ExplainableRecommender()
