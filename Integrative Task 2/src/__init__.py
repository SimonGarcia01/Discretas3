#Now here we do the whole pipeline

from preprocessing import preprocess_pipeline
from models import build_models
from training import train_all_models
from evaluate import evaluate_saved_models
from visualize import make_visualizations

import os
import pandas as pd

def run_pipeline(
	data_path: str = "data/cleaned_data.xlsx",
	max_words: int = 4000,
	max_len: int = 44,
	random_state: int = 42,
):

	# Prprocess first
	df_train, df_val, df_test = preprocess_pipeline(
		data_path,
		text_col="clean",
		target_col="y",
		max_words=max_words,
		max_len=max_len,
		test_size=0.3,
		val_size=0.5,
		random_state=random_state,
	)

	# Now save sequences dataset so the evalute.py can read it afterwards
	out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sequences_dataset.xlsx"))
	out_dir = os.path.dirname(out_path)
	os.makedirs(out_dir, exist_ok=True)
	with pd.ExcelWriter(out_path) as writer:
		df_train.to_excel(writer, sheet_name="df_train", index=False)
		df_val.to_excel(writer, sheet_name="df_val", index=False)
		df_test.to_excel(writer, sheet_name="df_test", index=False)

	# Build  the models
	dnn_model, rnn_model, lstm_model = build_models(max_words=max_words, input_length=max_len, embedding_dim=50)

	# Train the models with the datasets
	models_tuple, histories_tuple = train_all_models(df_train, df_val, dnn_model, rnn_model, lstm_model)

	# Now just evaluate the saved models
	evaluate_saved_models()
	
    #Missing the visualization step
	make_visualizations(models_tuple, histories_tuple)

	return models_tuple, histories_tuple


# Execute the pipeline at import time to match previous behaviour
run_pipeline()