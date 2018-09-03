from pythonds.basic.stack import Stack

s=Stack()

exp = raw_input("Enter an expression: ")
for i in exp:
	if i == '(':
		s.push(i)
	elif i ==')':
		try:
			s.pop()
		except IndexError:
			print("Parathenses don't match")
			quit()

if s.isEmpty():
	print("Expression is OK")
else:
	print("Parathenses don't match")