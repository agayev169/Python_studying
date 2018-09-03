import random

play_again = True

while play_again:
	def calcScore(player):
		aces = 0
		res = 0
		for i in player:
			try:
				res += int(i)
			except:
				if (i != "A"):
					res += 10
				else:
					aces += 1

		i = 0
		while i < aces:
			if res + aces * 11 < 21:
				res += aces * 11
				return res
			else:
				res += 1
			aces -= 1
		return res


	cards = ["2", "3", "4", "5", "6", "7", "8", 
			 "9", "10", "J", "Q", "K", "A"]

	print ("-" * 10 + "BlackJack" + "-" * 10)

	dealer = []
	dealer.append(random.choice(cards))
	print("Dealer's cads: " + str(dealer[0]) + ", *")

	player = []
	player.append(random.choice(cards))
	player.append(random.choice(cards))
	print("Your cards: " + ", ".join(player))
	print("Your score: " + str(calcScore(player)))

	choice = ""
	while choice != "stand" and choice != "pass":
		choice = input(("Your choice(hit/stand/pass): "))

		print("You chose " + choice)
		if choice == "pass":
			print("You lose! :c")
			quit()
		elif choice == "hit":
			player.append(random.choice(cards))
			print("Your cards: " + ", ".join(player))
			print("Your score: " + str(calcScore(player)))
		elif choice == "stand":
			print("Your cards: " + ", ".join(player))
			print("Your score: " + str(calcScore(player)))
		else:
			print("No choice called " + choice + ". Try again")
			pass

		if calcScore(player) > 21:
			print("You lose! :c")
			quit()

	while calcScore(dealer) < 17:
		dealer.append(random.choice(cards))


	print("Dealer's cads: " + ", ".join(dealer))
	print("Dealer's score: " + str(calcScore(dealer)))

	if calcScore(dealer) > 21:
		print("Dealer has more than 21.\nYou win! :3")
	elif calcScore(dealer) < calcScore(player):
		print("You win! :3")
	elif calcScore(dealer) == calcScore(player):
		print("Draw :/")
	else:
		print("You lose! :c")

	ans = '123'
	while ans != 'yes' and ans != 'no':
		ans = input("Do you want to play again?(yes/no) ")
		if ans == 'yes':
			play_again = True
		elif ans == 'no':
			play_again = False
		else:
			print("You have written a bullshit")
