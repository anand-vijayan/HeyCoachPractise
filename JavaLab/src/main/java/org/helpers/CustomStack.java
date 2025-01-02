package org.helpers;

public class CustomStack {
    private int[] stack;
    private int[] inc;  // Array to store increment values for each index
    private int size;   // Current size of the stack
    private int maxSize; // Maximum size of the stack

    /** Initialize the stack with a maximum size. */
    public CustomStack(int maxSize) {
        this.maxSize = maxSize;
        stack = new int[maxSize];
        inc = new int[maxSize];
        size = 0; // Start with an empty stack
    }

    /** Push element x onto stack. */
    public void push(int x) {
        if (size < maxSize) {
            stack[size] = x;
            inc[size] = 0;  // Initialize the increment value for the new element
            size++;
        }
    }

    /** Pop the top element and return it. */
    public int pop() {
        if (size == 0) {
            return -1;
        }

        int idx = size - 1;
        int val = stack[idx] + inc[idx]; // Apply any accumulated increment
        if (idx > 0) {
            inc[idx - 1] += inc[idx];  // Carry over the increment to the previous element
        }
        inc[idx] = 0; // Reset the increment for the popped element
        size--;

        return val;
    }

    /** Increment the bottom k elements by val. */
    public void increment(int k, int val) {
        // Increment only the first min(k, size) elements
        int limit = Math.min(k, size);
        if (limit > 0) {
            inc[limit - 1] += val;  // Add val to the k-th element's increment value
        }
    }
}
