import pandas as pd

def ready_df(df):
    """
    Specific to the 'perfumes_df_ready.csv'. 
    Cleans up the 'gender' variable.
    Adds dummies of 'designer', 'group', 'gender' to the dataframe, and drops original ones.
    Adds 'perfume_id' instead of 'perfume_name' to make all numeric dataframe

    Parameters: 
    -----------
    df: the data frame loaded from 'perfumes_df_ready.csv'.
    """
    # cleaning up the gender column in data
    gender = []
    for k in df['perfume_name']:
        gender.append(k.split('for')[1].lower().strip())


    for n, k in enumerate(gender):
        if ('her' in k[:4]) or ('women' in k[:]) and ('women and men' not in k):
            gender[n]='women'
        elif ('him' in k[:4]) or ('men' in k[:4]):
            gender[n] = 'men'
        elif ('women and men' in k):
            gender[n] = 'women and men'
        else:
            gender[n] = 'unknown'
            
    
    # replace the old with the new
    df['gender'] = gender
    
    # get dummies 
    dummies = pd.get_dummies(df[['designer', 'group', 'gender']])
    
    # add them to the dataframe
    new_df = pd.concat([df, dummies], axis = 1, sort = False)
    
    # add a quick perfume_id instead of perfume_name
    new_df['perfume_id'] = df.index 
    new_df.drop(['all_notes','top_notes', 'middle_notes','base_notes', 'synopsis', 'main_accords', 'perfume_name',
                'designer', 'group', 'gender'], axis = 1, inplace = True)
    
    return new_df

#------------------------------------------------

def classify_target(df):
    
    """
    Specific to the 'perfumes_df_ready.csv'.
    Bins the ratings into bins of width 0.5 by rounding up every ratings to the nearest .5
    Apply it to the cleaned data frame, with target column 'overall_rating'.

    Parameters:
    -----------
    df: the data frame loaded from 'perfumes_df_ready.csv' and/or cleaned first.
    """
    # make sure format is unified
    ratings_continuous = df['overall_rating'].map(lambda x: round(float(x), 2))
    
    # make target into bins of 0.5 width
    new_y = []
    for y in ratings_continuous:
        if (y > int(y)) & (y <= int(y) + .5):
            new_y.append(int(y) + .5)

        elif (y > int(y) + .5) & (y < int(y) + 1):
            new_y.append(int(y) + 1)

        elif (y == int(y)):
            new_y.append(y)
            
    # add the new ratings to the data frame; and remove old ones
    df['ratings_classes'] = new_y
    df.drop('overall_rating', axis = 1, inplace = True)
    
    return df

#-------------------------------------------------------------------

def make_frames(df):

    """
    Spesific to the data frame loaded from "perfumes_df_ready.csv"
    Creates and returns the training and testing data frames, in that order. 
    
    Parameters: 
    -----------
    df: the cleaned data frame, with its target classified into bins, using classify_target()
        function.
    """

    # the data frame of perfumes not having overall rating, that I want to predict
    test_df = df.loc[df['ratings_classes'] == -1, :]
    test_df.drop(['ratings_classes'], axis = 1, inplace = True)

    # The training data frame
    training_df = df.loc[df['ratings_classes']!=-1, :]

    return training_df, test_df

#----------------------------------------------------------------

def categorize_classes(df):
    """
    Run classify_target() before this one, or it won't work.
    This function's job to rename the bins of ratings previously created into whole integers, 
    so the predictions would work. 
    Highest ratings 5 is equal to 9, 8 is 4.5 and so on, down to 1 is 1.0
    {1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}

    Parameters:
    -----------
    df: the data frame loaded from 'perfumes_df_ready.csv'.
    """
    
    # make the dictionary to map the old values to the new categories
    classes_dict = {key: value for key, value in zip(sorted(list(df['ratings_classes'].value_counts().index)), 
                                             range(1,10))}
    
    # replace
    df['labels'] = df['ratings_classes'].map(classes_dict)
    
    return df   

#-----------------------------------------------------------------

