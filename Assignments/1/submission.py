import os
import subprocess
import monte_carlo_approach
import ZeroR
import perceptron

print("testing Monte Carlo")
MonteCarlo.monte_carlo_approach(10)

print("\ntesting ZeroR")
fname = "tennis.csv"
with open(fname) as f :
    data = f.readlines()
    print(f"zeroR: {ZeroR.zeroR(data)}")
    print(f"randR: {ZeroR.randR(data)}")


print("\ntesting wc")
os.system("python wc.py --strip --lower --nonwords --separator=',' C:/Users/ryanf/Programming/CS362/Assignments/1/tennis.csv")

print("\ntesting perceptron")
perceptron.perceptron_training()