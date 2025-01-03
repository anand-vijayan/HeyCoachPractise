package org.modules;

import org.dto.Edge;
import org.dto.GraphNode;
import org.dto.Line;
import org.helpers.DisjointSet;

import java.util.*;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@SuppressWarnings("ALL")
public class GraphAlgorithms {
    //region Variables & Constants
    private static final int[] rowDirs = {-1, 1, 0, 0};
    private static final int[] colDirs = {0, 0, -1, 1};
    static class Edge {
        int src, dest, weight;

        public Edge(int src, int dest, int weight)
        {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
    }

    // Defines subset element structure
    static class Subset {
        int parent, rank;

        public Subset(int parent, int rank)
        {
            this.parent = parent;
            this.rank = rank;
        }
    }
    //endregion

    //region BFS & DFS 1
    public static int WaterCubes(int[] height) {
        int left = 0; // Left pointer
        int right = height.length - 1; // Right pointer
        int maxArea = 0; // To track the maximum area

        // Two-pointer approach
        while (left < right) {
            // Calculate the current area
            int currentArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currentArea); // Update max area if the current one is larger

            // Move the pointer pointing to the shorter line inward
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return maxArea;
    }

    public static int WordLadder(String startWord, String targetWord, List<String> wordList) {
        // Check if the targetWord is in the word list
        if (!wordList.contains(targetWord)) {
            return 0;
        }

        // Initialize BFS structures
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        // Add start word to the queue and visited set
        queue.offer(startWord);
        visited.add(startWord);

        int steps = 1; // Start with step 1, as we start from the startWord

        // Begin BFS
        while (!queue.isEmpty()) {
            int size = queue.size();

            // Process all words at the current level (current step)
            for (int i = 0; i < size; i++) {
                String currentWord = queue.poll();

                // Try transforming the current word to all possible words
                char[] wordArray = currentWord.toCharArray();

                for (int j = 0; j < wordArray.length; j++) {
                    char originalChar = wordArray[j];

                    // Try all possible substitutions for character at position j
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == originalChar) continue; // Skip if no change

                        wordArray[j] = c;
                        String newWord = new String(wordArray);

                        // If we found the target word, return the current number of steps
                        if (newWord.equals(targetWord)) {
                            return steps + 1;
                        }

                        // If the new word is in the word list and not visited, add it to the queue
                        if (wordList.contains(newWord) && !visited.contains(newWord)) {
                            queue.offer(newWord);
                            visited.add(newWord);
                        }
                    }

                    // Restore the original character to continue the loop
                    wordArray[j] = originalChar;
                }
            }

            // Increment the steps after processing one level
            steps++;
        }

        // If we exhaust the queue without finding targetWord, return 0
        return 0;
    }

    public static int MinimumBroadcastTime(List<List<Integer>> nodes, int n, int k) {
        int[][] edges = GraphAlgorithms.getIntArray(nodes);
        // Step 1: Build the graph as an adjacency list
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            graph.put(i, new ArrayList<>());
        }

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int delay = edge[2];
            graph.get(u).add(new int[]{v, delay});
        }

        // Step 2: Initialize the distances array and priority queue (min-heap)
        int[] distances = new int[n + 1];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[k] = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[]{0, k});

        // Step 3: Run Dijkstra's algorithm
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int currentDistance = current[0];
            int node = current[1];

            // If the current distance is already greater than the recorded distance, skip it
            if (currentDistance > distances[node]) continue;

            // Relax the edges
            for (int[] neighbor : graph.get(node)) {
                int nextNode = neighbor[0];
                int delay = neighbor[1];

                int newDistance = currentDistance + delay;
                if (newDistance < distances[nextNode]) {
                    distances[nextNode] = newDistance;
                    pq.offer(new int[]{newDistance, nextNode});
                }
            }
        }

        // Step 4: Find the maximum time to reach all nodes
        int maxTime = (Arrays.stream(distances, 1, n + 1).max().isPresent())
                ? Arrays.stream(distances, 1, n + 1).max().getAsInt()
                : Integer.MAX_VALUE;

        // If any node is unreachable, return -1
        if (maxTime == Integer.MAX_VALUE) {
            return -1;
        }

        return maxTime;
    }

    public static boolean DeterminingPathValidity(List<List<Integer>> graph, int source, int destination) {
        int[][] edges = GraphAlgorithms.getIntArray(graph);
        int n = edges.length;

        // Step 1: Build the graph
        List<List<Integer>> graphOutput = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graphOutput.add(new ArrayList<>());
        }

        for (int[] edge : edges) {
            if(edge.length < 2) continue;
            graphOutput.get(edge[0]).add(edge[1]);
        }

        // Step 2: Arrays to track visited nodes and the recursion stack for cycle detection
        boolean[] visited = new boolean[n];
        boolean[] inRecursionStack = new boolean[n];

        // Step 3: DFS to check for cycles and validate paths
        boolean[] canReachDestination = new boolean[n];
        Arrays.fill(canReachDestination, false);
        dfs(graphOutput, source, visited, inRecursionStack, canReachDestination);

        // If there's a cycle involving the source node, return false
        if (inRecursionStack[source]) {
            return false;
        }

        // Check if all paths from source node lead exclusively to destination node
        return canReachDestination[source] && canReachDestination[destination];
    }

    public static List<String> ReconstructItinerary(List<List<String>> tickets) {
        String[][] ticketArray = getStringArray(tickets);
        // Step 1: Build the graph using a map where the key is the departure airport
        // and the value is a sorted list of destination airports
        Map<String, PriorityQueue<String>> graph = new HashMap<>();

        // Construct the graph
        for (String[] ticket : ticketArray) {
            graph.putIfAbsent(ticket[0], new PriorityQueue<>());
            graph.get(ticket[0]).offer(ticket[1]);
        }

        // Step 2: Define the result list and start the DFS from "MUM"
        List<String> itinerary = new ArrayList<>();
        dfs(graph, "MUM", itinerary);

        return itinerary;
    }
    //endregion

    //region BFS & DFS 2
    public static int ShortestPathInBinaryMatrix(int N, int M, int[][] A, int X, int Y) {
        // Edge case: if start or destination is blocked
        if (A[0][0] == 0 || A[X][Y] == 0) {
            return -1;
        }

        // BFS setup
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[N][M];
        queue.offer(new int[]{0, 0, 0}); // {row, col, distance}
        visited[0][0] = true;

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int row = current[0];
            int col = current[1];
            int dist = current[2];

            // If we reached the destination
            if (row == X && col == Y) {
                return dist;
            }

            // Explore the 4 possible directions
            for (int i = 0; i < 4; i++) {
                int newRow = row + rowDirs[i];
                int newCol = col + colDirs[i];

                // Check if the new position is within bounds and is walkable
                if (newRow >= 0 && newRow < N && newCol >= 0 && newCol < M && A[newRow][newCol] == 1 && !visited[newRow][newCol]) {
                    visited[newRow][newCol] = true;
                    queue.offer(new int[]{newRow, newCol, dist + 1});
                }
            }
        }

        // If we reach here, there's no path
        return -1;
    }

    public static boolean WaterAndJugProblem(int x, int y, int target) {
        if(target > x + y) return false;
        if(x == target || y == target || x + y == target) return true;
        Queue<int []> q = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        q.offer(new int[]{0,0});
        visited.add(0+","+0);

        while(!q.isEmpty()) {
            int [] cur = q.poll();
            int a = cur[0];
            int b = cur[1];

            if(a == target || b == target || a + b == target) {
                return true;
            }
            List<int[]> nextList = generateNext(a, b, x, y);
            for (int[] nextState : nextList) {
                int nexta = nextState[0];
                int nextb = nextState[1];

                String key = nexta + "," + nextb;
                if (visited.contains(key)) continue;
                q.offer(new int[]{nexta, nextb});
                visited.add(key);
            }
        }
        return false;
    }

    public static int BulbSwitches(int n, int presses) {

        int allBulbsOn = (1 << n) - 1;
        int button2Effect = 0;
        int button3Effect = 0;
        int button4Effect = 0;

        for (int i = 1; i < n; i += 2) {
            button2Effect |= (1 << i);
        }

        for (int i = 0; i < n; i += 2) {
            button3Effect |= (1 << i);
        }

        for (int i = 0; i < n; i++) {
            if ((i - 1) % 3 == 0) {
                button4Effect |= (1 << i);
            }
        }

        Set<Integer> currentStates = new HashSet<>();
        currentStates.add(allBulbsOn);

        for (int i = 0; i < presses; i++) {
            Set<Integer> nextStates = new HashSet<>();

            for (int state : currentStates) {
                nextStates.add(state ^ allBulbsOn);
                nextStates.add(state ^ button2Effect);
                nextStates.add(state ^ button3Effect);
                nextStates.add(state ^ button4Effect);
            }

            currentStates = nextStates;
        }

        return currentStates.size();
    }

    public static List<Integer> ShortestPaths(int n, int[][] graph) {
        List<Integer> result = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            result.add(findShortestCycle(n, graph, i));
        }

        return result;
    }

    public static boolean CloneGraph(GraphNode node) {
        GraphNode clonedGraph = cloneGraph(node);
        return validateGraphEquality(node, clonedGraph);
    }
    //endregion

    //region Minimum Spanning Trees
    public static List<Integer> WeakestRows(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;
        List<int[]> soldierCount = new ArrayList<>();

        for (int i = 0; i < m; i++) {
            int count = 0;
            for (int j = 0; j < n; j++) {
                count += mat[i][j];
            }
            soldierCount.add(new int[]{count, i});
        }

        soldierCount.sort((a, b) -> {
            if (a[0] == b[0]) {
                return Integer.compare(a[1], b[1]);
            }
            return Integer.compare(a[0], b[0]);
        });
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = soldierCount.get(i)[1];
        }

        return Arrays.stream(result).boxed().collect(Collectors.toList());
    }

    public static int FactorsOfKthLargest(int n, int k) {
        List<Integer> factors = new ArrayList<>();

        // Find all factors of n
        for (int i = 1; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                factors.add(i);  // i is a factor
                if (i != n / i) {
                    factors.add(n / i);  // n / i is also a factor
                }
            }
        }

        // Sort factors in ascending order
        Collections.sort(factors);

        // Check if there are enough factors for the kth largest
        if (k > factors.size()) {
            return -1;  // Not enough factors
        }

        // Return the kth largest factor
        return factors.get(factors.size() - k);

    }

    public static int FindingMinimumSpanningTree(int N, List<List<Integer>> edges) {

        List<org.dto.Edge> edgeList = new ArrayList<>();
        for (List<Integer> edge : edges) {
            if(edge.size() < 3) continue;
            edgeList.add(new org.dto.Edge(edge.get(0), edge.get(1), edge.get(2)));
        }

        // Sort edges by weight
        Collections.sort(edgeList);

        // Initialize disjoint set for Kruskal's algorithm
        DisjointSet ds = new DisjointSet(N);

        int mstWeight = 0;
        int edgeCount = 0;

        // Process edges in sorted order
        for (org.dto.Edge edge : edgeList) {
            if (edgeCount == N - 1) break; // MST has n-1 edges

            int rootU = ds.find(edge.src);
            int rootV = ds.find(edge.dest);

            // Include edge if it doesn't form a cycle
            if (rootU != rootV) {
                ds.union(rootU, rootV);
                mstWeight += edge.weight;
                edgeCount++;
            }
        }

        return mstWeight;
    }

    public static void RohansGame(Line[] lines, int n, int l) {
        // Sort the lines by length
        Arrays.sort(lines);

        // Initialize disjoint set for Kruskal's algorithm
        DisjointSet ds = new DisjointSet(n);

        List<int[]> mst = new ArrayList<>();

        // Process edges in sorted order
        for (Line line : lines) {
            int rootP1 = ds.find(line.p1);
            int rootP2 = ds.find(line.p2);

            // Include line if it doesn't form a cycle
            if (rootP1 != rootP2) {
                ds.union(rootP1, rootP2);
                mst.add(new int[]{line.p1, line.p2});
            }

            // Stop if we have n-1 edges in the MST
            if (mst.size() == n - 1) {
                break;
            }
        }

        for (int[] edge : mst) {
            System.out.println(edge[0] + " " + edge[1]);
        }
    }

    public static int[][] MinimumSpanningTree(int V, int m, int[][] input) {
        int j = 0;
        int noOfEdges = 0;
        List<Edge> edges = new ArrayList<>();

        for(int[] val : input) {
            edges.add(new Edge(val[0], val[1], val[2]));
        }

        // Allocate memory for creating V subsets
        Subset subsets[] = new Subset[V];

        // Allocate memory for results
        Edge results[] = new Edge[V];

        // Create V subsets with single elements
        for (int i = 0; i < V; i++) {
            subsets[i] = new Subset(i, 0);
        }

        // Number of edges to be taken is equal to V-1
        while (noOfEdges < V - 1) {

            // Pick the smallest edge. And increment
            // the index for next iteration
            Edge nextEdge = edges.get(j);
            int x = findRoot(subsets, nextEdge.src);
            int y = findRoot(subsets, nextEdge.dest);

            // If including this edge doesn't cause cycle,
            // include it in result and increment the index
            // of result for next edge
            if (x != y) {
                results[noOfEdges] = nextEdge;
                union(subsets, x, y);
                noOfEdges++;
            }

            j++;
        }

        // Print the contents of result[] to display the
        // built MST

        int minCost = 0;

        for (int i = 0; i < noOfEdges; i++) {
            minCost += results[i].weight;
        }

        int[][] output = new int[1][1];
        output[0][0] = minCost;
        return output;
    }
    //endregion

    //region Greedy
    //endregion

    //region Private Methods
    private static int[][] getIntArray(List<List<Integer>> list) {
        int[][] array = new int[list.size()][];

        for (int i = 0; i < list.size(); i++) {
            List<Integer> sublist = list.get(i);
            array[i] = new int[sublist.size()];

            for (int j = 0; j < sublist.size(); j++) {
                array[i][j] = sublist.get(j);
            }
        }
        return array;
    }

    private static String[][] getStringArray(List<List<String>> list) {
        String[][] array = new String[list.size()][];

        for (int i = 0; i < list.size(); i++) {
            List<String> sublist = list.get(i);
            array[i] = new String[sublist.size()];

            for (int j = 0; j < sublist.size(); j++) {
                array[i][j] = sublist.get(j);
            }
        }
        return array;
    }

    private static void dfs(List<List<Integer>> graph, int node, boolean[] visited, boolean[] inRecursionStack, boolean[] canReachDestination) {
        visited[node] = true;
        inRecursionStack[node] = true;

        // If this node can reach destination, mark it
        if (node == canReachDestination.length - 1) {
            canReachDestination[node] = true;
        }

        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfs(graph, neighbor, visited, inRecursionStack, canReachDestination);
            }
            // If the neighbor is in the recursion stack, a cycle is detected
            if (inRecursionStack[neighbor]) {
                canReachDestination[node] = true;
            }
        }

        inRecursionStack[node] = false;
    }

    private static void dfs(Map<String, PriorityQueue<String>> graph, String current, List<String> itinerary) {
        // While there are destinations from the current airport, continue the search
        while (graph.containsKey(current) && !graph.get(current).isEmpty()) {
            // Take the lexicographically smallest destination
            String next = graph.get(current).poll();
            // Recurse with the next destination
            dfs(graph, next, itinerary);
        }
        // Add the current airport to the itinerary (post-order)
        itinerary.add(0, current);
    }

    private static List<int[]> generateNext(int a, int b, int capA, int capB){
        List<int[]> result = new ArrayList<>();
        // only empty jug a;
        result.add(new int[]{0, b});

        // only empty jug b;
        result.add(new int[]{a, 0});

        // only fill up jug b;
        result.add(new int[]{a, capB});

        // only fill up jug a;
        result.add(new int[]{capA, b});

        // pour from jug a to fill up jub b;
        int pourAmt = Math.min(a, capB - b);
        result.add(new int[]{a - pourAmt, b + pourAmt});

        // pour from jug b to fill up jub a;
        pourAmt = Math.min(b, capA - a);
        result.add(new int[]{a +  pourAmt, b - pourAmt});

        return result;
    }

    private static int findShortestCycle(int n, int[][] graph, int start) {
        // Distance array to keep track of the distance from the start city
        int[] dist = new int[n];
        Arrays.fill(dist, -1);  // Initialize all distances to -1
        dist[start] = 0;

        // Queue for BFS: store (city, distance from start)
        Queue<Integer> queue = new LinkedList<>();
        queue.add(start);

        // Start BFS
        while (!queue.isEmpty()) {
            int city = queue.poll();

            // Explore all neighbors
            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (graph[city][neighbor] == 1) {  // There's a road from city to neighbor
                    if (dist[neighbor] == -1) {  // Not visited
                        dist[neighbor] = dist[city] + 1;
                        queue.add(neighbor);
                    } else if (neighbor == start) {  // Found a cycle
                        return dist[city] + 1;  // Found a cycle returning to the start
                    }
                }
            }
        }

        // If no cycle is found
        return -1;
    }

    private static GraphNode cloneGraph(GraphNode node) {
        if (node == null) {
            return null;
        }

        Map<GraphNode, GraphNode> visited = new HashMap<>();
        return dfs(node, visited);
    }

    private static GraphNode dfs(GraphNode node, Map<GraphNode, GraphNode> visited) {
        if (visited.containsKey(node)) {
            return visited.get(node);
        }

        GraphNode cloneNode = new GraphNode(node.data);
        visited.put(node, cloneNode);

        for (GraphNode neighbour : node.neighbours) {
            cloneNode.neighbours.add(dfs(neighbour, visited));
        }

        return cloneNode;
    }

    private static boolean validateGraphEquality(GraphNode original, GraphNode cloned) {
        if (original == null && cloned == null) {
            return true;
        }
        if (original == null || cloned == null) {
            return false;
        }

        Queue<GraphNode> originalQueue = new LinkedList<>();
        Queue<GraphNode> clonedQueue = new LinkedList<>();

        Set<GraphNode> visitedOriginal = new HashSet<>();
        Set<GraphNode> visitedCloned = new HashSet<>();

        originalQueue.offer(original);
        clonedQueue.offer(cloned);

        while (!originalQueue.isEmpty()) {
            GraphNode currentOriginal = originalQueue.poll();
            GraphNode currentCloned = clonedQueue.poll();

            if (currentOriginal.data != currentCloned.data) {
                return false;
            }

            if (visitedOriginal.contains(currentOriginal) || visitedCloned.contains(currentCloned)) {
                continue;
            }

            visitedOriginal.add(currentOriginal);
            visitedCloned.add(currentCloned);

            if (currentOriginal.neighbours.size() != currentCloned.neighbours.size()) {
                return false;
            }

            for (int i = 0; i < currentOriginal.neighbours.size(); i++) {
                GraphNode neighbourOriginal = currentOriginal.neighbours.get(i);
                GraphNode neighbourCloned = currentCloned.neighbours.get(i);

                if (!visitedOriginal.contains(neighbourOriginal) && !visitedCloned.contains(neighbourCloned)) {
                    originalQueue.offer(neighbourOriginal);
                    clonedQueue.offer(neighbourCloned);
                }
            }
        }

        return true;
    }

    private static int find(int x, int[] parent) {
        if (parent[x] != x) {
            parent[x] = find(parent[x], parent);
        }
        return parent[x];
    }

    private static void union(int x, int y, int[] parent, int[] rank) {
        int rootX = find(x, parent);
        int rootY = find(y, parent);

        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    private static void union(Subset[] subsets, int x,
                              int y)
    {
        int rootX = findRoot(subsets, x);
        int rootY = findRoot(subsets, y);

        if (subsets[rootY].rank < subsets[rootX].rank) {
            subsets[rootY].parent = rootX;
        }
        else if (subsets[rootX].rank
                < subsets[rootY].rank) {
            subsets[rootX].parent = rootY;
        }
        else {
            subsets[rootY].parent = rootX;
            subsets[rootX].rank++;
        }
    }

    // Function to find parent of a set
    private static int findRoot(Subset[] subsets, int i)
    {
        if (subsets[i].parent == i)
            return subsets[i].parent;

        subsets[i].parent
                = findRoot(subsets, subsets[i].parent);
        return subsets[i].parent;
    }
    //endregion
}
