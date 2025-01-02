package org.sessiontests;

import org.dto.TreeNode;
import org.mindmaps.Graphs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Eight {
    //region Private Methods
    private static void inorderHelper(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        inorderHelper(root.left, result);  // Traverse the left subtree
        result.add(root.val);  // Visit the current node
        inorderHelper(root.right, result);  // Traverse the right subtree
    }

    private static List<Integer> inOrderTraversalHelper(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        inorderHelper(root, result);
        return result;
    }
    //endregion

    public static long ContinuousNeighborhoods(int[] numbers) {
        // Sort the array first
        Arrays.sort(numbers);
        int n = numbers.length;
        int start = 0;
        int count = 0;

        // Sliding window approach
        for (int end = 0; end < n; end++) {
            // Shrink the window if the condition is violated
            while (numbers[end] - numbers[start] > 2) {
                start++;
            }
            // Count the number of sub-arrays in the current valid window
            count += (end - start + 1);
        }

        return count;
    }

    public static void inorderTraversal(TreeNode root) {
        List<Integer> inorder = inOrderTraversalHelper(root);
        System.out.println(inorder);
    }

    public static int FindTheTownJudge(int N, int[][] trust){
        return Graphs.FindTheTownJudge(N, trust);
    }
}
