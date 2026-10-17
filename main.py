import pandas as pd


def main():
    df = pd.read_csv("heart_clean.csv")
    df.info()


if __name__ == "__main__":
    main()
