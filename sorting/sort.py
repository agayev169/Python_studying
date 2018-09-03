import random
import time

# arr = [random.randint(0, 100000) for _ in range(10000)]
arr = [x for x in range(10, 0, -1)]
# arr = [random.randint(0, 9) for _ in range(10)]

def bubble(arr):
	for i in range(len(arr)):
		for j in range(len(arr) - i - 1):
			if arr[j] > arr[j + 1]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]

	return arr

def quick(arr, begin = 0, end = len(arr)):
	if end - begin < 2:
		return arr
	mid = (end - begin) // 2 + begin
	left = begin
	right = end - 1
	while left < right:
		while arr[left] < arr[mid] and left < end:
			left += 1
		while arr[right] > arr[mid] and right > begin:
			right -= 1
		if left < right:
			arr[left], arr[right] = arr[right], arr[left]
			left += 1
			right -= 1
	quick(arr, begin, mid)
	quick(arr, mid, end)
	return arr

def insertion(arr):
	

# t = time.clock_gettime(0);


print(arr)
# arr = quick(arr)
# arr = bubble(arr)
# print(arr)


# print("Time spent for sorting:", (time.clock_gettime(0) - t), "seconds")
