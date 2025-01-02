package org.helpers;

import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
    private Queue<Integer> queue1;
    private Queue<Integer> queue2;

    /** Initialize your data structure here. */
    public MyStack() {
        queue1 = new LinkedList<>();
        queue2 = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        // Push the new element into queue1
        queue1.offer(x);
    }

    /** Removes the element on the top of the stack and returns it. */
    public int pop() {
        // Transfer elements from queue1 to queue2, except for the last one (which is the top of the stack)
        while (queue1.size() > 1) {
            queue2.offer(queue1.poll());
        }
        // The last element in queue1 is the top of the stack
        int topElement = queue1.poll();

        // Swap the roles of queue1 and queue2, so queue1 becomes the main queue for the next operation
        Queue<Integer> temp = queue1;
        queue1 = queue2;
        queue2 = temp;

        return topElement;
    }

    /** Get the top element. */
    public int top() {
        // Transfer elements from queue1 to queue2, except for the last one (which is the top of the stack)
        while (queue1.size() > 1) {
            queue2.offer(queue1.poll());
        }
        // The last element in queue1 is the top of the stack
        int topElement = queue1.peek();

        // Push the top element back into queue2
        queue2.offer(queue1.poll());

        // Swap the roles of queue1 and queue2
        Queue<Integer> temp = queue1;
        queue1 = queue2;
        queue2 = temp;

        return topElement;
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue1.isEmpty();
    }
}
