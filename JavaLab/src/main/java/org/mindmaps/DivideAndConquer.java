package org.mindmaps;

import org.dto.TreeNode;

public class DivideAndConquer {

    public static TreeNode ConvertSortedArrayToBinarySearchTree(int[] numbers) {
        return helper(numbers, 0, numbers.length - 1);
    }

    public static int MajorityElement(int[] numbers) {
        int candidate = numbers[0];
        int count = 1;

        // Boyer-Moore Voting Algorithm
        for (int i = 1; i < numbers.length; i++) {
            if (count == 0) {
                candidate = numbers[i];
                count = 1;
            } else if (numbers[i] == candidate) {
                count++;
            } else {
                count--;
            }
        }

        return candidate;
    }

    public static int MaximumSubArray(int[] numbers) {
        // Initialize current_sum and max_sum
        int current_sum = numbers[0];
        int max_sum = numbers[0];

        // Iterate through the array starting from the second element
        for (int i = 1; i < numbers.length; i++) {
            // Update current_sum: either extend the subarray or start a new one
            current_sum = Math.max(numbers[i], current_sum + numbers[i]);
            // Update max_sum with the maximum of the previous max_sum and current_sum
            max_sum = Math.max(max_sum, current_sum);
        }

        // Return the largest sum found
        return max_sum;
    }

    public static int ReverseBits(int n) {
        int result = 0;  // The result that will store the reversed bits.

        // Loop through all 32 bits of the number
        for (int i = 0; i < 32; i++) {
            // Shift the result to the left to make space for the new bit
            result <<= 1;

            // Extract the least significant bit of n and add it to result
            result |= (n & 1);

            // Shift n to the right to process the next bit
            n >>= 1;
        }

        return result;  // Return the reversed integer
    }

    public static TreeNode MaximumBinaryTree(int[] numbers) {
        return buildTree(numbers, 0, numbers.length - 1);
    }

    //region Private Methods

    private static TreeNode helper(int[] numbers, int left, int right) {
        // Base case: if the left index is greater than the right, return null
        if (left > right) {
            return null;
        }

        // Find the middle index
        int mid = left + (right - left) / 2;

        // Create the root node with the middle value
        TreeNode root = new TreeNode(numbers[mid]);

        // Recursively build the left and right subtrees
        root.left = helper(numbers, left, mid - 1);   // Left subtree
        root.right = helper(numbers, mid + 1, right);  // Right subtree

        return root;
    }

    private static TreeNode buildTree(int[] nums, int left, int right) {
        if (left > right) {
            return null;  // Base case: no elements to process
        }

        // Find the maximum value in the current range [left, right]
        int maxIndex = findMaxIndex(nums, left, right);

        // Create a node with the maximum value
        TreeNode node = new TreeNode(nums[maxIndex]);

        // Recursively build the left and right subtrees
        node.left = buildTree(nums, left, maxIndex - 1);
        node.right = buildTree(nums, maxIndex + 1, right);

        return node;
    }

    private static int findMaxIndex(int[] nums, int left, int right) {
        int maxIndex = left;
        for (int i = left; i <= right; i++) {
            if (nums[i] > nums[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    //endregion
}
