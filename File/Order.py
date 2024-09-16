

class Product:
    def __init__(self,I,P):
        self._productID = I
        self._productPrice = P

    def getOrderItemID(self,n):
        return self._productID

    def getOrderItemPrice(self,n):
        return self._productPrice


class OrderStatus():
    def __init__(self, N, D):
        self._HasShipped = False


class Order(OrderStatus,Product):
    def __init__(self,N,D):
        OrderStatus.__init__(self,N,D)
        self._ordernumber = N
        self._date = D
        self._product_list = []
        self._Num_ordered = 0
        self._status = "NotShipped"
    def OrderItem(self,m):
        self._product_list.append(m)
        self._Num_ordered += 1

    def getOrderStatus(self):
        if self._Num_ordered > 0:
            for i in range(self._Num_ordered):
                self._HasShipped = True
                self._status = "HasShipped"
                return self._status
            else:
                return self._status

    def numberofitemsordered(self):
        return self._Num_ordered

    def getOrderItemID(self,n):
        return self._product_list[n-1]._productID


    def getOrderItemPrice(self,n):
        return self._product_list[n-1]._productPrice

product1 = Product("beans", 0.45)
product2 = Product("eggs", 1.25)

myOrder = Order(1, "1/1/17")
myOrder.OrderItem(product1)
myOrder.OrderItem(product2)
print(myOrder.getOrderStatus())
print(myOrder.getOrderItemID(1))
print(myOrder.getOrderItemPrice(1))
print(myOrder.getOrderItemID(2))
print(myOrder.getOrderItemPrice(2))
