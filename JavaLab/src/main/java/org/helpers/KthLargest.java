package org.helpers;

import java.util.PriorityQueue;

public class KthLargest {
    private PriorityQueue<Integer> minHeap; // Min-heap to store the top k elements
    private int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        minHeap = new PriorityQueue<>(k); // Initialize the min-heap with a capacity of k

        // Add all elements in nums to the heap
        for (int num : nums) {
            add(num);
        }
    }

    public int add(int val) {
        minHeap.offer(val); // Add the new value to the heap

        // If the size of the heap exceeds k, remove the smallest element
        if (minHeap.size() > k) {
            minHeap.poll();
        }

        // The smallest element in the heap is the kth largest
        return minHeap.peek();
    }
}
