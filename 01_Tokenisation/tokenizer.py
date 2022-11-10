import sys

line = sys.stdin.readline()

while line!='':
    
    line = line.replace('.',' .')
    line = line.replace(',',' ,')
    line = line.replace(';',' ;')
    line = line.replace(':',' :')
    line = line.replace(' ','\n')

    sys.stdout.write(line)
    line = sys.stdin.readline()

#Punctuations are considered seperate tokens
