num1 = float(raw_input("Enter first number: "))
operator = raw_input("Enter operator: ")
num2 = float(raw_input("Enter second number: "))

if operator == '+':
	result = num1 + num2
elif operator == '-':
	result = num1 - num2
elif operator == '*':
	result = num1 * num2
elif operator == '/':
	if num2 == 0:
		print "Division by zero is impossible"
		exit()
	result = num1 / num2

print result
