package org.sessiontests;

import org.dto.Node;
import org.dto.TrieNode;
import org.modules.AdvancedDataStructure;

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Vector;

public class Seven {
    public static int GoldNuggets(Vector<Integer> nums) {
        // Calculate initial total value and target reduction
        double totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        double targetReduction = totalSum / 2.0;

        // Priority queue (max-heap) for managing nugget values by largest first
        PriorityQueue<Double> maxHeap = new PriorityQueue<>((a, b) -> Double.compare(b, a));
        for (int num : nums) {
            maxHeap.offer((double) num);
        }

        double currentReduction = 0;
        int splits = 0;

        // Continue splitting until we've reduced the total sum by at least half
        while (currentReduction < targetReduction) {
            // Take the largest nugget, split it, and calculate the reduction
            double largest = maxHeap.poll();
            double halved = largest / 2.0;
            currentReduction += halved;

            // Reinsert the halved nugget back into the max heap
            maxHeap.offer(halved);
            splits++;
        }

        return splits;
    }

    public static ArrayList<Integer> SortTheNodesInBST(Node root) {
        return AdvancedDataStructure.SortTheNodesInBST(root);
    }

    public static long LogQueryInterface(TrieNode root, String s) {
        TrieNode node = root;
        for (char ch : s.toCharArray()) {
            int index = ch - 'a'; // Calculate index for the character
            if (node.children[index] == null) {
                return 0; // If prefix is not found, return 0
            }
            node = node.children[index];
        }
        return node.count; // Return the count at the end of the prefix
    }
}
