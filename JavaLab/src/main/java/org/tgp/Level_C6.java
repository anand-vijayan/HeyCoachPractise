package org.tgp;

import java.util.*;

public class Level_C6 {
    public int Tribonacci(int n) {
        // Handle base cases
        if (n == 0) {
            return 0;
        } else if (n == 1 || n == 2) {
            return 1;
        }

        // Initialize base values
        int t0 = 0, t1 = 1, t2 = 1;

        // Compute Tribonacci numbers iteratively
        for (int i = 3; i <= n; i++) {
            int tNext = t0 + t1 + t2;
            t0 = t1;
            t1 = t2;
            t2 = tNext;
        }

        return t2;

    }

    public int[] NextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        Arrays.fill(result, -1); // Initialize result array with -1
        Stack<Integer> stack = new Stack<>(); // Stack to hold indices

        // Traverse the array twice
        for (int i = 0; i < 2 * n; i++) {
            int currentIndex = i % n;
            // Process stack for current element
            while (!stack.isEmpty() && nums[stack.peek()] < nums[currentIndex]) {
                int index = stack.pop();
                result[index] = nums[currentIndex];
            }
            // Push the index onto the stack if in the first pass
            if (i < n) {
                stack.push(currentIndex);
            }
        }

        return result;

    }

    public List<Integer> HelpStudents(List<Integer> marks, int n) {
        List<Integer> result = new ArrayList<>();
        // Initialize the result with -1
        for (int i = 0; i < n; i++) {
            result.add(-1);
        }
        Stack<Integer> stack = new Stack<>(); // Stack to store indices

        for (int i = 0; i < n; i++) {
            // Process the stack for the current student
            while (!stack.isEmpty() && marks.get(stack.peek()) > marks.get(i)) {
                int index = stack.pop();
                result.set(index, marks.get(i));
            }
            // Push the current index onto the stack
            stack.push(i);
        }

        return result;

    }
}
