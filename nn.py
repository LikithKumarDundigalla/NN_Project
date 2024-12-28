import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}


def train(model_path="model", train_path="data/train.csv", dev_path="data/dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow)

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16)

    # Building a Convolutional Neural Network (CNN) model for sequence classification.
    # The model starts with an Embedding layer to convert token indices into dense vector representations of fixed size (128-dimensional embeddings).
    # This is followed by a Conv1D layer with 128 filters and a kernel size of 5, applying ReLU activation to extract local features from sequences.
    # A GlobalMaxPooling1D layer is added to reduce the dimensionality by selecting the maximum value for each feature map, providing global feature representation.
    # Dropout layers with a 50% rate are included after the convolution and dense layers to prevent overfitting during training.
    # A fully connected Dense layer with 128 neurons and ReLU activation serves as a hidden layer for further feature transformation.
    # The output layer is a Dense layer with sigmoid activation, making it suitable for multi-label classification tasks by predicting probabilities for each label.

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=95),
        tf.keras.layers.Conv1D(filters=95, kernel_size=5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(95, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(labels), activation='sigmoid')
    ])

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])

    # fit the model to the training data, monitoring F1 on the dev data
    # ModelCheckpoint:Saves the model weights to the specified file path ("model_path.keras") whenever the validation F1 score improves.
    # Monitors the "val_f1_score" metric in "max" mode, meaning it looks for the highest value.
    # Ensures that only the best-performing model based on validation F1 score is saved.

    # EarlyStopping: Stops training early if no improvement in "val_f1_score" is observed for 3 consecutive epochs.
    # Prevents unnecessary computation and reduces overfitting by halting when the model stops improving.

    # ReduceLROnPlateau:Reduces the learning rate by a factor of 0.5 if the validation F1 score does not improve for 2 consecutive epochs.
    # Helps the optimizer converge more effectively by lowering the learning rate when progress slows.

    model.fit(
        train_dataset,
        epochs=10,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path+".keras",
                monitor="val_f1_score",
                mode="max",
                save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_score",
                mode="max",
                patience=3),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_f1_score",
                mode="max",
                factor=0.5,
                patience=2,
        )])


def predict(model_path="model", input_path="data/dev.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path+".keras")

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("data/submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
