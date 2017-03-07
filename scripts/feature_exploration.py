import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import cPickle as pickle

train = pd.read_json('../data/train.json')

print train.head()

def create_numerical_features(df):
	df['num_photos'] = df['photos'].apply(len) # how many photos a listing has
	df['num_features'] = df['features'].apply(len) # number of features listed
	df['num_description'] = df['description'].apply(lambda x: len(x.split(" "))) # description length
	df["created"] = pd.to_datetime(df["created"])
	df["created_year"] = df["created"].dt.year
	df["created_month"] = df["created"].dt.month
	df["created_day"] = df["created"].dt.day
	df["created_day_of_week"] = df["created"].dt.dayofweek


def create_text_features(df, type):
	# df : pandas dataframe
	# type : string, either test or train depending on which file it is

	# Features column

	df['features'] = df['features'].apply(lambda x: " ".join(x)) # combine list to a string

	stop = stopwords.words('english') # remove stopwords
	vect = CountVectorizer(stop_words=stop, max_features=200)
	features_dtm = vect.fit_transform(df['features'])

	with open('../processed/' + type +'_features_dtm.dat','wb') as outfile:
	    pickle.dump(features_dtm, outfile, pickle.HIGHEST_PROTOCOL)

	###########################################################################
	# Description column
	vect2 = CountVectorizer(stop_words=stop, max_features=200)
	description_dtm = vect2.fit_transform(df['description'])

	with open('../processed/' + type +'_description_dtm.dat','wb') as outfile:
	    pickle.dump(description_dtm, outfile, pickle.HIGHEST_PROTOCOL)	

	###########################################################################
	# Display address column
	vect3 = CountVectorizer(stop_words=stop, max_features=200)
	display_address_dtm = vect3.fit_transform(df['display_address'])

	with open('../processed/' + type +'_display_address_dtm.dat','wb') as outfile:
	    pickle.dump(display_address_dtm, outfile, pickle.HIGHEST_PROTOCOL)

def create_categorical_features(df):
	le = LabelEncoder()

	df.building_id = le.fit_transform(df.building_id)

	df.manager_id = le.fit_transform(df.manager_id)

	# used different method to encode interest level to preserve increasing order
	target_num = {'low':0, 'medium':1, 'high':2}
	df.interest_level = df.interest_level.apply(lambda x: target_num[x])

create_text_features(train, 'train')