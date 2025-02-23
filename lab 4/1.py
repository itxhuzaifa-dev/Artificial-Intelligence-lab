class Stack:
    def __init__(self):
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.stack.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.stack[-1]

    def size(self):
        return len(self.stack)

    def __str__(self):
        return str(self.stack)


stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print("Stack:", stack)
print("Popped item:", stack.pop())
print("Stack after pop:", stack)
print("Top item:", stack.peek())
print("Is stack empty?", stack.is_empty())
print("Stack size:", stack.size())
