package org.tgp;

import org.dto.ListNode;
import java.util.*;

public class Level_C5 {

    public static List<List<Integer>> CriticalConnection(int n, List<List<Integer>> connections) {
        // Step 1: Build the graph using an adjacency list
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (List<Integer> connection : connections) {
            int u = connection.get(0), v = connection.get(1);
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        // Step 2: Initialize discovery and low-link arrays
        int[] discovery = new int[n];
        int[] low = new int[n];
        Arrays.fill(discovery, -1); // -1 indicates that the node has not been visited
        Arrays.fill(low, -1);

        // Step 3: List to store critical connections
        List<List<Integer>> criticalConnections = new ArrayList<>();

        // Step 4: Perform DFS from node 0
        dfs(0, -1, discovery, low, graph, criticalConnections);

        return criticalConnections;
    }

    public static ListNode OddEvenLinkedList(ListNode head) {
        if (head == null || head.next == null) {
            return head; // No reordering needed for 0 or 1 node
        }

        ListNode odd = head;         // Pointer for odd nodes
        ListNode even = head.next;  // Pointer for even nodes
        ListNode evenHead = even;   // Save the start of the even list

        while (even != null && even.next != null) {
            odd.next = even.next;   // Link the next odd node
            odd = odd.next;         // Move the odd pointer

            even.next = odd.next;   // Link the next even node
            even = even.next;       // Move the even pointer
        }

        odd.next = evenHead; // Attach the even list at the end of the odd list
        return head;
    }

    public static int StarGraph(int[][] edges) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();

        // Count the frequency of each node
        for (int[] edge : edges) {
            frequencyMap.put(edge[0], frequencyMap.getOrDefault(edge[0], 0) + 1);
            frequencyMap.put(edge[1], frequencyMap.getOrDefault(edge[1], 0) + 1);
        }

        // Find the node with frequency equal to edges.length
        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            if (entry.getValue() == edges.length) {
                return entry.getKey();
            }
        }

        // Should not reach here for a valid star graph
        throw new IllegalArgumentException("Invalid star graph");
    }

    //region Private Methods
    private static int time = 0;
    private static void dfs(int node, int parent, int[] discovery, int[] low, List<List<Integer>> graph, List<List<Integer>> criticalConnections) {
        discovery[node] = low[node] = ++time; // Initialize discovery and low-link values

        for (int neighbor : graph.get(node)) {
            if (neighbor == parent) {
                continue; // Skip the edge leading back to the parent
            }

            if (discovery[neighbor] == -1) { // If the neighbor is not visited
                dfs(neighbor, node, discovery, low, graph, criticalConnections);

                // Update low-link value after visiting the neighbor
                low[node] = Math.min(low[node], low[neighbor]);

                // Check if the edge is a critical connection (bridge)
                if (low[neighbor] > discovery[node]) {
                    criticalConnections.add(Arrays.asList(node, neighbor));
                }
            } else {
                // Update low-link value for a back edge
                low[node] = Math.min(low[node], discovery[neighbor]);
            }
        }
    }
    //endregion
}
