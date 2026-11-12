
import matplotlib.pyplot as plt
import pandas as pd

from src.data_processing import load_and_split_dataset


def explore_dataset(dataset):

    
    train_df = dataset["train"].to_pandas()

    
    print("\n--- Dataset Information ---")

    print("Shape:", train_df.shape)

    print("\nColumns:")
    print(train_df.columns.tolist())

    print("\nSample records:")
    print(train_df.head())

    
    print("\n--- Missing Values ---")

    print(train_df.isnull().sum())

    
    print("\n--- Class Distribution ---")

    class_counts = train_df["label"].value_counts()

    print(class_counts)

    print("\nClass percentages:")
    print(train_df["label"].value_counts(normalize=True) * 100)

    
    class_counts.plot(kind="bar")

    plt.title("SMS Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Messages")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    
    train_df["message_length"] = (
        train_df["sms"].str.split().apply(len)
    )

    print("\n--- Message Length Statistics ---")

    print(
        train_df.groupby("label")["message_length"].describe()
    )

    
    train_df.boxplot(
        column="message_length",
        by="label",
        grid=False,
        showfliers=False
    )

    plt.title("Message Length by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Number of Words")
    plt.tight_layout()
    plt.show()

    
    print("\n--- Sample HAM Messages ---")

    print(
        train_df[train_df["label"] == 0]["sms"].head(5)
    )

    print("\n--- Sample SPAM Messages ---")

    print(
        train_df[train_df["label"] == 1]["sms"].head(5)
    )

    return train_df


if __name__ == "__main__":

    data = load_and_split_dataset()

    explore_dataset(data)
