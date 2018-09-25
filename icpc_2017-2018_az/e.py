import random
import time

def to_int(nums):
	nums_int = []
	for num in nums:
		nums_int.append(int(num))
	return nums_int

'''
def lcm(num1, num2):
	if num1 == num2:
		return num1
	if num1 > num2 and num1 % num2 == 0:
		return num1
	elif num2 > num1 and num2 % num1 == 0:
		return num2
	sm = num1 if num1 < num2 else num2
	for i in range(2, sm):
		if num1 % i == 0 and num2 % i == 0:
			num1 //= i
	return num1 * num2
'''

def gcd(num1, num2):
	if num1 % num2 == 0:
		return num1
	if num2 % num1 == 0:
		return num2
	if is_prime(num1) or is_prime(num2):
		return num1 * num2
	while num1 != num2:
		if num1 > num2:
			num1 -= num2
		else:
			num2 -= num1
	return num1

	'''
	if num1 == num2:
		return num1
	if num1 > num2:
		return gcd(num1 - num2, num2)
	return gcd(num1, num2 - num1)
'''

def lcm(num1, num2):
	return num1 * num2 // gcd(num1, num2)

def is_prime(num):
	for div in range(2, num // 2 + 1):
		if num % div == 0:
			return False
	return True

inputs = []
results = []

# n = int(input())
n = 100000
# nums = input().split(' ')
# nums = to_int(nums)

# t = time.clock_gettime(0)

# lcm(1, 10000000)

# print(time.clock_gettime(0) - t)

nums = [random.randint(1, 100) for _ in range(n)]

t = time.clock_gettime(0)

lcm_num = nums[0]
for num in nums:
	if not [lcm_num, num] in inputs:
		lcm_num_n = lcm(lcm_num, num)
	else:
		index = inputs.index([lcm_num, num])
		lcm_num_n = results[index]

	if lcm_num != lcm_num_n:
		lcm_num = lcm_num_n
		results.clear()
		inputs.clear()

	results.append(lcm_num_n)
	inputs.append([lcm_num, num])

print(time.clock_gettime(0) - t)

n_opers = 0
for num in nums:
	if lcm_num != num:
		n_opers += 1

print(n_opers)
print(time.clock_gettime(0) - t)
