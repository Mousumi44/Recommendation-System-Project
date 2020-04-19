#full_recommender.py
import pandas as pd
from math import sqrt
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.spatial import distance
from set_recommender import *

class FullRecommender():
    def __init__(self):
        self.preprocessing()
        self.content_recommender()

    def preprocessing(self):
        #Storing the movie information into a pandas dataframe
        movies_df = pd.read_csv('ml-latest/movies.csv')

        #Using regular expressions to find a year stored between parentheses
        #We specify the parantheses so we don't conflict with movies that have years in their titles
        movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Removing the parentheses
        movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
        #Removing the years from the 'title' column
        movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
        #Applying the strip function to get rid of any ending whitespace characters that may have appeared
        movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

        #Every genre is separated by a | so we simply have to call the split function on |
        movies_df['genres'] = movies_df.genres.str.split('|')

        #Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
        moviesWithGenres_df = movies_df.copy()

        #For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
        for index, row in movies_df.iterrows():
            for genre in row['genres']:
                moviesWithGenres_df.at[index, genre] = 1

        self.movies_df = movies_df
        #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
        self.moviesWithGenres_df = moviesWithGenres_df.fillna(0)

    def content_recommender(self):
        temp_inputMovies = get_interests()
        #print(temp_inputMovies)
        temp_inputMovies = temp_inputMovies.reset_index(drop=True)
        self.inputMovies = temp_inputMovies.drop('genres', 1)

        print('Creating Genre Table...')
        userGenreTable = self.createGenreTable(temp_inputMovies)
        genre_table_size = len(userGenreTable.index)

        #
        just_genres = userGenreTable.drop('title', 1).drop('genres', 1).drop('rating', 1)

        avg_rating = {}
        tag_ratings = {}
        self.vector_dict = {}
        print('Calculating Ratings...')

        for i in range(genre_table_size):
            vector = str(list(just_genres.iloc[i])).strip('[]')
            rating = float(self.inputMovies['rating'][i])
            self.vector_dict[vector] = i

            if bool(avg_rating) == False or vector not in avg_rating.keys():
              tag_ratings[vector] = []
              tag_ratings[vector].append(rating)
              avg_rating[vector] = rating
            else:
              tag_ratings[vector].append(rating)
              avg_rating[vector] = sum(tag_ratings[vector])/len(tag_ratings[vector])

            #print(avg_rating)
        self.userGenreTable = userGenreTable
        self.avg_rating = avg_rating
        self.find_highest_rated_tag()

    #
    def find_highest_rated_tag(self):
        print('Finding highest rated tag...')
        # checking for highest rated tag_set
        highest_rated_tag = max(self.avg_rating.items(), key= operator.itemgetter(1))[0]
        self.highest_rated_tag = highest_rated_tag
        self.hr_idx = self.vector_dict[highest_rated_tag]

        rem_avg_rating = self.avg_rating
        rem_avg_rating.pop(highest_rated_tag)

        cos_ = {}
        for vect in rem_avg_rating.keys():
            highest_rated_tag_list = [float(i) for i in highest_rated_tag.split(',')]
            vect_list = [float(i) for i in vect.split(',')]
            cos_[vect] = distance.cosine(highest_rated_tag_list, vect_list)

        # cosine similarity, finds vector most similar to the highest rated vector
        closest_vect = min(cos_.items(), key= operator.itemgetter(1))[0]
        self.closest_vect = closest_vect
        self.highest_rated_tag_list = highest_rated_tag_list

        #
        print('Generating Explanation...')
        self.generate_explanation()

    #
    def generate_explanation(self):
        closest_vect_rating = self.avg_rating[self.closest_vect]

        #convert to list of numbers
        closest_vect_list = [float(i) for i in self.closest_vect.split(',')]

        #
        vect_diffs = np.array(self.highest_rated_tag_list) - np.array(closest_vect_list)
        diff_columns = [idx for idx in range(len(vect_diffs)) if vect_diffs[idx] != 0]
        #
        vect_sim = np.array(self.highest_rated_tag_list) + np.array(closest_vect_list)
        sim_columns = [idx for idx in range(len(vect_sim)) if vect_sim[idx] != 0]

        genre1 = self.userGenreTable.columns[sim_columns[0]]

        for col in diff_columns:
            genre2 = self.userGenreTable.columns[col]

            if closest_vect_rating < 3:
                self.statement = "you don't like " + genre1 + " if it is a " + genre2
                user_pref_vect = np.array(self.highest_rated_tag_list)

            elif closest_vect_rating >= 3 and closest_vect_rating < 4:
                self.statement = "you like " + genre1 + " if it is a " + genre2
                user_pref_vect = vect_sim

            elif closest_vect_rating >= 4:
                self.statement = "you like " + genre1 + " especially if it is a " + genre2
                user_pref_vect = vect_sim

        self.user_pref_vect = user_pref_vect

        #
        print('Generating recommendation...')
        self.generate_recommendation()
    #
    def generate_recommendation(self):
        user_df = pd.DataFrame(self.user_pref_vect, index=['Adventure',	'Animation',	'Children',	'Comedy',	'Fantasy',	'Romance',	'Drama',	'Action',	'Crime',	'Thriller',	'Horror',	'Mystery',	'Sci-Fi',	'IMAX',	'Documentary',	'War',	'Musical',	'Western',	'Film-Noir',	'(no genres listed)'])
        user_df_T = user_df.T

        self.vector_dict.pop(self.highest_rated_tag)
        droppings = self.vector_dict.values()
        input_movies = self.inputMovies['rating'].drop(droppings)
        input_movies.iloc[self.hr_idx] = float(input_movies.iloc[self.hr_idx])
        userProfile = user_df_T.transpose().dot(input_movies)

        #Now let's get the genres of every movie in our original dataframe
        genreTable = self.moviesWithGenres_df.set_index(self.moviesWithGenres_df['movieId'])
        #And drop the unnecessary information
        genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

        recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
        recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

        rec = self.movies_df.loc[self.movies_df['movieId'].isin(recommendationTable_df.head(1).keys())]
        recommendation = rec.loc[rec.index[0]]['title']
        full_statement = "we recommend \'"+ recommendation + "\' because " + self.statement

        print(full_statement)

    #
    def createGenreTable(self, temp_inputMovies):
        temp_inputMovies.drop('rating', 1)
        genre_list = self.moviesWithGenres_df.columns.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

        for j in range(len(genre_list)):
            genre = genre_list[j]
            count_list = []
            for i in range(len(temp_inputMovies)):
                if genre in temp_inputMovies.iloc[i]['genres']:
                    count_list.append(1.0)
                else:
                    count_list.append(0.0)
            temp_inputMovies[genre] = count_list
        return temp_inputMovies

if __name__ == "__main__":
    FullRecommender()
