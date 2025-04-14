import numpy

file = open("file.txt", "r")
lines = file.readlines()
file.close()

for line in lines:
    fields = line.split(" ")
    
    print(fields)

