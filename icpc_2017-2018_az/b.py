formula = input()

def get_dist_vars(all_vars):
	duo = False
	dist_vars = [var for var in all_vars if len(var) == 1]

	for i in range(len(dist_vars) - 1, -1, -1):
		if dist_vars.count(dist_vars[i]) > 1:
			dist_vars.pop(i)

	for var in all_vars:
		if len(var) == 2:
			if var[1] not in dist_vars:
				dist_vars.append(var[1])
			else:
				duo = True
	return dist_vars, duo




all_vars = formula.split('|')
dist_vars = get_dist_vars(all_vars)

n_trues = 2 ** len(dist_vars[0]) if dist_vars[1] == True else 2 ** len(dist_vars[0]) - 1
print(n_trues)