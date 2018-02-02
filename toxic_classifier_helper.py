from math import log
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

VOCAB_SIZE = 25000
MAX_TEXT_LEN = 100
EMBEDDING_DIM = 128
MODEL_DIR = "saved_models"
SUBMISSION_DIR = "submissions"
TEST_CSV = "test.csv"


class ToxicClassifierHelper:
    def __init__(self, vocab_size=VOCAB_SIZE, max_text_len=MAX_TEXT_LEN, model_dir=MODEL_DIR,
                 submission_dir=SUBMISSION_DIR, train_csv="train.csv", submission_csv="test.csv",
                 embedding_dim=EMBEDDING_DIM):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_text_len = max_text_len
        self.model_dir = model_dir
        self.submission_dir = submission_dir
        self.train_df, self.raw_submission_df = self.load_training_and_test_data(train_csv,
                                                                                 submission_csv)
        self.initialize_tokenizer()
        self.X_submit = self.create_submission_data()

    def load_training_and_test_data(self, train_csv, submission_csv):
        train_df = pd.read_csv(train_csv)
        assert len(train_df) == 95851, "Training csv does not contain 95,851 records"
        submission_df = pd.read_csv(submission_csv)
        assert len(submission_df) == 226998, "Test csv does not contain 226,998 records"
        return train_df, submission_df

    def initialize_tokenizer(self):
        if len(self.train_df) == 0:
            self.load_train_df()

        comment_text = self.train_df["comment_text"]
        self.tokenizer.fit_on_texts(comment_text)

    def create_train_test_split(self, test_size):
        X = self.create_padded_tokens(self.train_df)
        y = self.create_y_classes(self.train_df)

        return train_test_split(X, y, test_size=test_size)

    def create_submission_data(self):
        submission_df = self.create_padded_tokens(self.raw_submission_df)
        submission_df.set_index("id", inplace=True)
        return submission_df

    def create_padded_tokens(self, df):
        comment_text = df["comment_text"].astype(str)
        self.tokenizer.fit_on_texts(comment_text)

        tokens = self.tokenizer.texts_to_sequences(comment_text)
        padded_tokens = pad_sequences(tokens)
        return padded_tokens

    def create_y_classes(self, df):
        y_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        y = df[y_classes].values
        return y

    def save_model(self, model, model_name):
        save_file = os.path.join(self.model_dir, model_name)
        model.save(save_file)
        print("Model saved to:%s" % save_file)

    def create_submission(self, model, submission_name):
        preds = model.predict(self.X_submit)
        return preds
        pred_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        submission = pd.DataFrame(colnames=pred_columns)
        submission[pred_columns] = preds

        assert submission.shape == (226998, 5), "Submission shape is not (226998, 6)"

        submission_file = os.path.join(self.submission_dir, submission_name)
        submission.to_csv(submission_file)
        print("Submission file created at %s" % submission_file)


def _bin_log_loss(pred, actual, eps=.000001):
    pred = eps if pred == 0 else pred
    pred = 1 - eps if pred == 1 else pred
    return actual * log(pred) + (1 - actual) * log(1 - pred)


def _log_loss_6(preds, actual, eps=.000001):
    preds = [pred or eps for pred in preds]
    losses = [_bin_log_loss(preds[x], actual[x]) for x in range(len(preds))]
    return -sum(losses) / len(losses)


def log_loss(preds, actual, eps=.000001):
    log_loss_all = [_log_loss_6(preds[x], actual[x]) for x in range(len(preds))]
    return sum(log_loss_all) / len(log_loss_all)
