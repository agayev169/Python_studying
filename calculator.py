num1 = int(raw_input("Enter first number: "))
operator = raw_input("Enter an operation: ")
num2 = int(raw_input("Enter second number: "))
if operator == "+":
	print str(num1) + " + " + str(num2) + " = " + str(num1 + num2)
elif operator == "-":
	print str(num1) + " - " + str(num2) + " = " + str(num1 - num2)
elif operator == "*":
	print str(num1) + " * " + str(num2) + " = " + str(num1 * num2)
elif operator == "/":
	print str(num1) + " / " + str(num2) + " = " + str(num1 / num2)