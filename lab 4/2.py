class Queue:
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, value):
        self.queue.append(value)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.queue.pop(0)

    def top(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.queue[0]

    def size(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)


# Example usage


queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print("Queue:", queue)
print("Dequeued item:", queue.dequeue())
print("Queue after dequeue:", queue)
print("Front item:", queue.top())
print("Is queue empty?", queue.is_empty())
print("Queue size:", queue.size())
