class queue:
    def __init__(self, capacity=1000):
        if not isinstance(capacity, int):
            print("Capacity must be an integer")
            return

        self.__capacity = capacity
        self.__content = []

    def is_empty(self):
        return len(self.__content) == 0

    def is_full(self):
        return len(self.__content) == self.__capacity

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full")
            return
        self.__content.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.__content.pop(0)

    def front(self):
        if not self.is_empty():
            return self.__content[0]


if __name__ == "__main__":
    queue = queue(capacity=5)
    queue.enqueue(1)
    queue.enqueue(2)
    print(queue.is_full())
  
    print(queue.front())
    print(queue.dequeue())

    print(queue.is_empty())
