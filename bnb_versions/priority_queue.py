import heapq

# Priority queue class for Node objects
class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, node):
        # Negate the objValue to simulate a max-heap
        heapq.heappush(self.heap, (-node.objValue, node))

    def pop(self):
        # Return the Node with the largest objValue
        if not self.heap:
            raise IndexError("pop from empty priority queue")
        return heapq.heappop(self.heap)[1]  # Extract the Node object

    def is_empty(self):
        return len(self.heap) == 0
