li = [1,2,3,4,5,2.2,3.2,4]
file1 = open('msg.txt','r')
for line in file1:
    for i in li:
        if i in line:
            print(i)
print(file1)