# Does not work properly!
inp = input()

nums_str = inp.split(' ')
nums = [int(x) for x in nums_str]
tmp = 0
# print(nums)
for x in nums:
	tmp += 2 ** x
# print(tmp)
nums.clear()
nums.append(tmp)
opers = []

newe1 = max(nums) // 4
newe2 = max(nums) - newe1
nums.append(newe1)
nums.append(newe2)
opers.append([max(nums), newe1, newe2])
nums.remove(max(nums))
count = 1
while(len(nums) > 0):
	if nums[0] == 1 and len(nums) == 1:
		break
	print(nums)
	maxe = max(nums)
	# print(maxe)
	newe1 = maxe // 2
	newe2 = maxe - newe1
	nums.append(newe1)
	nums.append(newe2)
	# print(nums)
	nums.remove(maxe)
	opers.append([maxe, newe1, newe2])
	count += 1
	for x in nums:
		if nums.count(x) > 1 or x == 0 or x == 1:
			nums.remove(x)
	if count > 10:
		quit()
print(count)
print(opers)
