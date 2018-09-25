symbols = {chr(x): x for x in range(128)}
count = 128

filename_in = 'test.txt'
f1 = open(filename_in, 'r')
filename_out = 'out.txt'
f2 = open(filename_out, 'w')



str_tmp = f1.read(1)
while True:
    c = f1.read(1)
    if c == '':
        break
    if str_tmp + c in symbols:
        str_tmp += c
        continue
    f2.write(str(symbols[str_tmp]) + ' ')
    symbols[str_tmp + c] = count
    count += 1
    str_tmp = c

f2.write(str(symbols[str_tmp]))