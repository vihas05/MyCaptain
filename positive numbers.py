positive=[]
print("To find postive number")
n=int(input("Enter the range:"))
i=1
while i<=n:
    a=int(input("Enter the number:"))
    positive.append(a)
    i+=1
print("The list is ",positive)
for a in positive:
    if(a<0):
        positive.remove(a)
print("Positive list is ",positive)







