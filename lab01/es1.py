file = open("file.txt", "r")
lines = file.readlines()
file.close()
participants = []

def isfloat(number):
    try:
        float(number)
        return True 
    except ValueError:
        return False

for line in lines:
    fields = line.split(" ")
    scores = []
    for word in line:
        if isfloat(word):
            scores.append(float(word))
    participants.append((fields[0], fields[1], scores))
