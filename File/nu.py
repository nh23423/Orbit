
word = list(input(':'))
print(word)
for i in range(len(word)):
    if ord(word[i]) >= ord('a') and ord(word[i]) <= ord('z'):
        print(1)

print('string\n')
