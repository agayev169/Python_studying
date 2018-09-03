import random

class Door:
	opened = False

	def __init__(self, prize):
		self.prize = prize

	def __str__(self):
		return str(self.opened) + " " + str(self.prize)

wins = 0
total = 0
for _ in range(100000):
	doors = []
	index = random.randint(0, 2)
	for i in range(3):
		doors.append(Door(i == index))

	index = random.randint(0, 2)
	for i in range(3):
		if doors[i].prize == False and doors[i].opened == False and doors[i].prize == False and i != index:
			doors[i].opened = True
			break

	for i in range(3):
		if doors[i].opened == False and i != index:
			index = i
			break

	if doors[index].prize == True:
		wins += 1

	total += 1

print(str(wins / total))