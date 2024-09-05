package org.sessiontests;

import java.util.*;

public class Four {
    public static String GramAndTheString(String s){
        // Track the last occurrence of each character
        int[] lastIndex = new int[26];
        for (int i = 0; i < s.length(); i++) {
            lastIndex[s.charAt(i) - 'a'] = i;
        }

        // Boolean array to check if a character is in the current result
        boolean[] inResult = new boolean[26];

        // Use a stack to build the result
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {
            char currentChar = s.charAt(i);

            // If the character is already in the result, skip it
            if (inResult[currentChar - 'a']) {
                continue;
            }

            // Maintain lexicographical order and remove duplicates
            while (!stack.isEmpty() && stack.peek() > currentChar && lastIndex[stack.peek() - 'a'] > i) {
                char removedChar = stack.pop();
                inResult[removedChar - 'a'] = false; // Mark it as not in result
            }

            // Add the current character to the stack
            stack.push(currentChar);
            inResult[currentChar - 'a'] = true; // Mark it as in result
        }

        // Build the result from the stack
        StringBuilder result = new StringBuilder();
        for (char c : stack) {
            result.append(c);
        }

        return result.toString();
    }

    public static int TrafficRush(String dirs) {
        Stack<Character> stack = new Stack<>();
        int collisions = 0;

        for (char car : dirs.toCharArray()) {
            if (car == 'R') {
                stack.push(car);
            } else if (car == 'L') {
                // Handle collisions with right-moving cars
                while (!stack.isEmpty() && stack.peek() == 'R') {
                    collisions += 2;  // Head-on collision
                    stack.pop();      // Right-moving car stops
                }
                // Handle collisions with stopped cars
                if (!stack.isEmpty() && stack.peek() == 'S') {
                    collisions++;    // Rear-end collision
                }
                // Add left-moving car to the stack (it will stop future cars)
                stack.push(car);
            } else if (car == 'S') {
                // Handle rear-end collisions with right-moving cars
                while (!stack.isEmpty() && stack.peek() == 'R') {
                    collisions++;    // Rear-end collision
                    stack.pop();     // Right-moving car stops
                }
                // Add stopped car to the stack
                stack.push(car);
            }
        }

        return collisions;
    }

    public static int GroupGuests(int[] nums) {
        int n = nums.length;
        int[] frequency = new int[4]; // Assuming groups are 1, 2, 3

        // Count occurrences of each group
        for (int num : nums) {
            frequency[num]++;
        }

        // Sort the array based on group and then guest index
        Arrays.sort(nums);

        // Calculate the minimum number of swaps
        int swaps = 0;
        int expectedIndex = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                swaps++;
            }
            expectedIndex += frequency[i + 1];
        }

        return swaps;
    }
}
