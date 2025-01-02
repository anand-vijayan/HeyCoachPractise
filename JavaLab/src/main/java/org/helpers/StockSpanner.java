package org.helpers;

import java.util.Stack;

public class StockSpanner {

    private Stack<int[]> stack;

    /** Initialize the StockSpanner object. */
    public StockSpanner() {
        stack = new Stack<>();
    }

    /** Returns the span of the stock's price for the current day. */
    public int next(int price) {
        int span = 1;  // The span of today's price is at least 1 (itself).

        // While there are elements in the stack and the current price is greater than
        // or equal to the price at the top of the stack, pop elements and accumulate spans.
        while (!stack.isEmpty() && stack.peek()[0] <= price) {
            span += stack.pop()[1];  // Add the span of the popped element.
        }

        // Push the current price and its calculated span onto the stack.
        stack.push(new int[]{price, span});

        return span;
    }
}
