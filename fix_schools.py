

fp = open("schools.txt", "r")
line = fp.read()
parts = line.split(",", 408)
first = parts[0]
rest = ",".join(parts[1:])
rest = rest.replace(",", ",\n\t")
result = first + "," + rest
print(result)
fp.close()
