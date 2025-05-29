import pandas as pd
from decision_tree import five_fold

# Demo decision tree
data = pd.read_csv('NPHA-doctor-visits.csv')
attr_dict = {'Number of Doctors Visited': [1, 2, 3],
             'Age': [1, 2],
             'Physical Health': [-1, 1, 2, 3, 4, 5],
             'Mental Health': [-1, 1, 2, 3, 4, 5],
             'Dental Health': [-1, 1, 2, 3, 4, 5, 6],
             'Employment': [-1, 1, 2, 3, 4],
             'Stress Keeps Patient from Sleeping': [0, 1],
             'Medication Keeps Patient from Sleeping': [0, 1],
             'Pain Keeps Patient from Sleeping': [0, 1],
             'Bathroom Needs Keeps Patient from Sleeping': [0, 1],
             'Unknown Keeps Patient from Sleeping': [0, 1],
             'Trouble Sleeping': [-1, 1, 2, 3],
             'Prescription Sleep Medication': [-1, 1, 2, 3],
             'Race': [-2, -1, 1, 2, 3, 4, 5],
             'Gender': [-2, -1, 1, 2]
             }
f1_scores = five_fold(data, 'Number of Doctors Visited', attr_dict)
print(f"Doctor Visits F1 Scores: {f1_scores}")
