import pandas as pd
import matplotlib.pyplot as plt


def question1(bc):
    print(bc["Class"])


def question2(bc):
    print(bc["Class"].mode())


def question3(bc):
    print(bc[bc["Class"] == "no-recurrence-events"][["Age", "Menopause"]].mode())


def question4(bc):
    bc[bc["Class"] == "recurrence-events"]["Age"].value_counts().sort_index().plot(
        kind="bar"
    )
    plt.show()


def main():
    bc = pd.read_csv(
        "breast-cancer.data",
        names=[
            "Class",
            "Age",
            "Menopause",
            "Tumor Size",
            "Inv Nodes",
            "Node Caps",
            "Degree",
            "Breast",
            "Quadrant",
            "Irradiated?",
        ],
    )
    question1(bc)
    question2(bc)
    question3(bc)
    question4(bc)


if __name__ == "__main__":
    main()
