import pandas as pd
from datetime import datetime
from tqdm import tqdm
import random


def comments_timerange(comments, timestamps, days = 365):
    '''Function that keeps only comments (sort by 'created utc' in descending order)  within a certain timerange (in days)'''
    
    # convert from timestamp utc to datetime format
    most_recent_timestamp = datetime.utcfromtimestamp(timestamps[0])
    
    # initialize the stop values of the while loop
    k = 0
    time_range = 0

    while (time_range < days) and (k < len(timestamps)):
        # compute the timerange between the most recent comment and the comment k
        time_range = (most_recent_timestamp - datetime.utcfromtimestamp(timestamps[k])).days
        k += 1

    return comments[:k-2], timestamps[:k-2]


def generate_feeds(df, nb_feeds = 10, nb_words_per_feed = 500, exact = True, days = 365, seed = 0):
    ''' Function to create k feeds with n words for each author in df. Only the comments within a timerange (specified in day units) are kept.
    If exact = False, the last comment will be keep entirely even though the nb words is over the limite. 
    If an author doesn't have enought comments to fill n_feeds, this author won't appear in the returned dataframe  '''
    
    df_grouped = df.groupby('author')
    
    dict = {
        'author': [],
        'timerange': [],
    }

    # Create two columns per file: contents and words per comment
    for i in range(nb_feeds):
        filename   = 'feed' + str(i+1)
        slice_name = 'slices' + str(i+1)
        dict[filename] = []
        dict[slice_name] = []
    
    for name, group in tqdm(df_grouped):

        # Take all comments and timestamp for the author
        comments = group['body'].to_list()
        timestamp = group['created_utc'].to_list()

        # Keep only comments written within a year
        comments, timestamp = comments_timerange(comments, timestamp, days)
        if len(comments) == 0:
            continue
        # Randomize the comments and the timestamp similarly
        c = list(zip(comments, timestamp))
        random.Random(seed).shuffle(c)
        comments, timestamp = zip(*c)

        # initialize the return value
        i = 0
        files = []
        words_per_comment = []

        # for loop to create the k files
        for j in range(nb_feeds):
            timestamp_keep = []
            file = []
            n_words = []
            words = 0
            while (words < nb_words_per_feed) and (i<len(comments)):
                # Append the comment i to the file
                file += comments[i].split()

                # Store the number of words of comment i in an array
                n_words.append(len(comments[i].split()))
                
                # Count the number of words in the file
                words += len(comments[i].split())

                # Timestamps of kept comments
                timestamp_keep.append(timestamp[i])

                i += 1
            # if exact, the end of the last comment is truncated to reach exactly 500 words 
            if exact:
                file = file[:nb_words_per_feed]
                n_words[-1] = n_words[-1] - (len(file) - nb_words_per_feed)
            files.append(file)
            words_per_comment.append(n_words)
            
        # We save the author and their files only if we managed to complete all the files
        if i != len(comments):
            # Store all the data in the dictionnary
            dict['author'].append(name)
            dict['timerange'].append((datetime.utcfromtimestamp(max(timestamp_keep))- datetime.utcfromtimestamp(min(timestamp_keep))).days)
            
            # Store all the files and words per comment in separate columns
            for i in range(nb_feeds):
                filename   = 'feed' + str(i+1)
                slice_name = 'slices' + str(i+1)
                dict[filename].append(files[i])
                dict[slice_name].append(words_per_comment[i])
    
    df_out = pd.DataFrame.from_dict(dict).set_index('author')

    return df_out