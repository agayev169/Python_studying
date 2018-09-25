import random

suits = ['Hearts', 'Spades', 'Clubs', 'Diamonds']
cards = []

for card in ['Two', 'Three', 'Four', 'Five', 'Six', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']:
	for suit in suits:
		cards.append(str(card) + " of " + suit)

def shuffle():
	new_cards = []
	while len(cards) > 0:
		card = random.choice(cards)
		new_cards.append(card)
		cards.remove(card)
	return new_cards

cards = shuffle()

for card in cards:
	if cards.index(card) % 2 == 0:
		print('%18s' % card, end = '')
	else:
		print('%25s' % card)