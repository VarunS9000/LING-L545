import sys

line = sys.stdin.readline()

while line!='':
    for c in ['. ','? ','! ']:
        line = line.replace(c[0]+' ',c[0]+'\n')

    sys.stdout.write(line)
    line = sys.stdin.readline()
