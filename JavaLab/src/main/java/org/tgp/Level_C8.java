package org.tgp;

import org.dto.Item;
import org.dto.TrieNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Level_C8 {
    public static int LongestIncreasingSubsequence(int[] arr, int n) {
        if (n == 0) return 0;

        // dp[i] will store the length of the LIS ending at index i
        int[] dp = new int[n];
        // Initialize each dp value to 1 since the smallest LIS ending at any element is the element itself
        Arrays.fill(dp, 1);

        // Fill the dp array
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                // If arr[i] is greater than arr[j], we can extend the LIS ending at arr[j]
                if (arr[i] > arr[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        // The answer is the maximum value in the dp array
        int maxLength = 0;
        for (int i = 0; i < n; i++) {
            maxLength = Math.max(maxLength, dp[i]);
        }

        return maxLength;
    }

    public int Encoder(String[] words) {
        TrieNode root = new TrieNode();

        // Step 1: Insert words into the Trie
        for (String word : words) {
            TrieNode node = root;
            // Traverse the word in reverse order to build the Trie with suffixes
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                node.childrenMap.putIfAbsent(c, new TrieNode());
                node = node.childrenMap.get(c);
            }
            node.isEndOfWord = true;
        }

        // Step 2: Calculate the minimum length of the encoding
        int result = 0;
        for (String word : words) {
            TrieNode node = root;
            boolean isUniqueSuffix = false;
            // Traverse the word in reverse order to find the shortest contribution
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                node = node.childrenMap.get(c);
                if (node.isEndOfWord) {
                    isUniqueSuffix = true;
                    break;
                }
            }
            if (isUniqueSuffix) {
                result += word.length() + 1; // Add 1 for the '#' character
            }
        }
        return result;
    }

    public static double FractionalKnapsack(int N, int W, List<Integer> values, List<Integer> weights) {
        List<Item> items = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            items.add(new Item(values.get(i), weights.get(i)));
        }

        // Sort items based on value-to-weight ratio in descending order
        items.sort((a, b) -> Double.compare(b.ratio, a.ratio));

        double totalValue = 0.0;  // To store the total value of items in the knapsack
        int remainingCapacity = W;  // The remaining capacity of the knapsack

        for (int i = 0; i < N; i++) {
            if (remainingCapacity == 0) {
                break;  // If the knapsack is full, stop
            }

            // If the current item can be fully included in the knapsack
            if (items.get(i).weight <= remainingCapacity) {
                totalValue += items.get(i).value;
                remainingCapacity -= items.get(i).weight;
            } else {
                // Otherwise, take the fraction of the item
                totalValue += items.get(i).value * ((double) remainingCapacity / items.get(i).weight);
                remainingCapacity = 0;  // The knapsack is now full
            }
        }

        return totalValue;
    }
}
