#Here I just repeated wha'ts in the preprocessing.ipynb file

import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

######################################
# This is how to use this pipeline:
# from src.preprocessing import preprocess_pipeline
# df_train, df_val, df_test = preprocess_pipeline('../data/cleaned_data.xlsx') 
#######################################

#Added some parameters to make it more flexible
def preprocess_pipeline(
	data_path,
	text_col="clean",
	target_col="y",
	max_words=4000,
	max_len=44,
	test_size=0.3,
	val_size=0.5,
	random_state=42,
):

	# Load the data
	df = pd.read_excel(data_path, sheet_name=0, header=0)

	# Check if the columns are actually there
	if text_col not in df.columns or target_col not in df.columns:
		raise ValueError(f"Expected columns '{text_col}' and '{target_col}' in the data")

	# Keep the clean text and the target column
	df = df[[text_col, target_col]].copy()

	#Saw the dupliates in the notebook so i'll drop them here too
	df = df.drop_duplicates().reset_index(drop=True)

	# I know it happened in our case so i'll just do it anyway
	df = df[df[text_col].notnull()].reset_index(drop=True)

	# Now do the first split
	texts = df[text_col]
	labels = df[target_col]

	texts_train, texts_temp, y_train, y_temp = train_test_split(
		texts, labels, test_size=test_size, random_state=random_state
	)

	# Now second split for the test and validation sets
	texts_val, texts_test, y_val, y_test = train_test_split(
		texts_temp, y_temp, test_size=val_size, random_state=random_state
	)

	# Now the tokenization and padding
	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(texts_train)

	seq_train = tokenizer.texts_to_sequences(texts_train)
	seq_val = tokenizer.texts_to_sequences(texts_val)
	seq_test = tokenizer.texts_to_sequences(texts_test)

	X_train = pad_sequences(seq_train, maxlen=max_len)
	X_val = pad_sequences(seq_val, maxlen=max_len)
	X_test = pad_sequences(seq_test, maxlen=max_len)

	# Now just make each into df's and add the labels
	df_train = pd.DataFrame(X_train)
	df_val = pd.DataFrame(X_val)
	df_test = pd.DataFrame(X_test)

	df_train["y"] = y_train.reset_index(drop=True)
	df_val["y"] = y_val.reset_index(drop=True)
	df_test["y"] = y_test.reset_index(drop=True)

	return df_train, df_val, df_test