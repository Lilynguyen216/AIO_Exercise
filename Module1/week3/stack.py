class stack:
    def __init__(self, capacity=1000):
        if not isinstance(capacity, int):
            print("Capacity must be an integer")
            return

        self.__capacity = capacity
        self.__content = []

    def push(self, item):
        if self.is_full():
            print("Stack is full")
            return
        self.__content.append(item)

    def pop(self):
        if not self.is_empty():
            return self.__content.pop()

    def is_empty(self):
        return len(self.__content) == 0

    def is_full(self):
        return len(self.__content) == self.__capacity

    def top(self):
        if not self.is_empty():
            return self.__content[-1]


if __name__ == "__main__":
    stack = stack(capacity=5)
    stack.push(1)
    stack.push(2)
    print(stack.is_full())

    print(stack.top())
    print(stack.pop())
    print(stack.top())
    print(stack.pop())

    print(stack.is_empty())
