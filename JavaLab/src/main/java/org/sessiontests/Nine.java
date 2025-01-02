package org.sessiontests;

import java.util.*;

public class Nine {

    //region Private Methods
    private static boolean bfs(List<List<Integer>> graph, int start, int[] colors) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        colors[start] = 0; // Start coloring with color 0

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : graph.get(node)) {
                if (colors[neighbor] == -1) {
                    // Color the neighbor with the opposite color
                    colors[neighbor] = 1 - colors[node];
                    queue.offer(neighbor);
                } else if (colors[neighbor] == colors[node]) {
                    // If the neighbor has the same color, the graph is not bipartite
                    return false;
                }
            }
        }

        return true;
    }

    private static void dfs(List<List<Integer>> graph, boolean[] visited, int node, List<Integer> component) {
        visited[node] = true;
        component.add(node);

        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfs(graph, visited, neighbor, component);
            }
        }
    }

    private static boolean isCompleteComponent(List<Integer> component, List<List<Integer>> graph) {
        int size = component.size();
        // A complete graph of size k should have exactly k * (k - 1) / 2 edges
        int expectedEdges = size * (size - 1) / 2;
        int actualEdges = 0;

        // Count the actual number of edges within this component
        Set<Integer> componentSet = new HashSet<>(component);  // to check if the edge belongs to the component
        for (int node : component) {
            for (int neighbor : graph.get(node)) {
                if (componentSet.contains(neighbor) && node < neighbor) {  // Count each edge once
                    actualEdges++;
                }
            }
        }

        return actualEdges == expectedEdges;
    }
    //endregion

    public static int MinimumEvacuationTime(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int freshCount = 0;

        // Traverse the grid to initialize the queue with rotten oranges and count fresh oranges
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[] {i, j, 0}); // (row, col, time)
                } else if (grid[i][j] == 1) {
                    freshCount++;
                }
            }
        }

        // If there are no fresh oranges, return 0 since there's nothing to rot
        if (freshCount == 0) {
            return 0;
        }

        // Directions for up, down, left, right
        int[] dirs = {-1, 0, 1, 0, -1};
        int time = 0;

        // Perform BFS
        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            int x = curr[0], y = curr[1], t = curr[2];
            time = Math.max(time, t);

            for (int i = 0; i < 4; i++) {
                int nx = x + dirs[i];
                int ny = y + dirs[i + 1];

                // Check if the new position is within bounds and has a fresh orange
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    grid[nx][ny] = 2; // Turn it rotten
                    freshCount--; // Decrease the count of fresh oranges
                    queue.offer(new int[] {nx, ny, t + 1}); // Add it to the queue with incremented time
                }
            }
        }

        // If there are still fresh oranges left, return -1
        return freshCount == 0 ? time : -1;

    }

    public static boolean IsolatedOrInterconnected(List<List<Integer>> graph) {
        int n = graph.size();
        int[] colors = new int[n];
        Arrays.fill(colors, -1); // -1 means unvisited

        // BFS to check bipartite property for each unvisited node (handles disconnected components)
        for (int i = 0; i < n; i++) {
            if (colors[i] == -1) { // if this node hasn't been colored
                if (!bfs(graph, i, colors)) {
                    return false; // if one of the components is not bipartite
                }
            }
        }

        return true; // if all components are bipartite
    }

    public static int CountTheNumberOfCompleteComponents(int n, List<List<Integer>> edges) {
        // Step 1: Build the graph
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }

        for (List<Integer> edge : edges) {
            graph.get(edge.get(0)).add(edge.get(1));
            graph.get(edge.get(1)).add(edge.get(0));
        }

        // Step 2: Find connected components using DFS
        boolean[] visited = new boolean[n];
        int completeComponentsCount = 0;

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                // Step 3: Get the connected component starting from node i
                List<Integer> component = new ArrayList<>();
                dfs(graph, visited, i, component);

                // Step 4: Check if this component is complete
                if (isCompleteComponent(component, graph)) {
                    completeComponentsCount++;
                }
            }
        }

        return completeComponentsCount;
    }
}
