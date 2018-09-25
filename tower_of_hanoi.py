moves = []

def solve(from_t, buf_t, to_t, n_disks):
	global n
	if n_disks > 0:
		n += 1
		solve(from_t, to_t, buf_t, n_disks - 1)
		moves.append(str(from_t) + ' -> ' + str(to_t))
		solve(buf_t, from_t, to_t, n_disks - 1)


from_t = 0
buf_t = 1
to_t = 2

for n_disks in range(20):
	n = 0
	solve(from_t, buf_t, to_t, n_disks)
	print(n)