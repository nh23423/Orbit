'''
class Queue:

    def __init__(self,maxItems):
        self.items = [None]* maxItems
        self.front = 0
        self.rear = -1
        self.maxsize = maxItems
        self.size = 0
        self.empty = True
        self.full = False

    def isFull(self):
        if self.maxsize == self.size:
            self.full = True

    def isEmpty(self):
        if self.size == 0:
            self.empty = True

    def add(self,n):
        if self.full:
            print('Queue is full.')
        else:
            self.items.append(n)
            self.rear = (self.rear+1)%self.maxsize
            self.size += 1

    def remove(self):
        if self.empty:
            print('Queue is empty')
        else:
            self.items.remove(self.items[0])
            self.front = (self.front+1)%self.maxsize
            self.size -= 1
            return self.items[0]

class pQueue:

    def __init__(self,maxItems):
        self.items = [None]*maxItems
        self.front = 0
        self.rear = -1
        self.maxsize = maxItems
        self.size = 0
        self.empty = True
        self.full = False

    def isFull(self):
        if self.maxsize == self.size:
            self.full = True

    def isEmpty(self):
        if self.size == 0:
            self.empty = True

    def remove(self):
        if self.empty:
            print('Queue is empty')
        else:
            self.items.pop(0)
            self.front = (self.front+1)%self.maxsize
            self.size -= 1
            return self.items[0]

    def add(self,n):
        if self.full:
            print('Queue is full.')
        else:
            if self.size>0:
               for i in range(self.size):
                   if self.items[i]>n:
                    self.items.insert(i,n)
               self.rear = (self.rear + 1) % self.maxsize
               self.size += 1
            else:
               self.items.append(n)
               self.rear = (self.rear + 1) % self.maxsize
               self.size += 1





p = pQueue(6)
print(p.items)
'''

arr = [112,4,2,81,827,1930,91,221]

def mergesort(arr):
    if len(arr) > 1:

        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        mergesort(L)
        mergesort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()








