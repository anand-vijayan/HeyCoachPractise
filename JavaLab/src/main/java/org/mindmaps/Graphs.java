package org.mindmaps;

import org.helpers.UnionFind;

import java.util.*;

public class Graphs {

    //region Variables & Constants
    private static boolean[] visited;
    private static int count = 0;
    private static List<List<Integer>> list;
    private static int n;
    //endregion

    //region Simple Graph
    public static int FindCenterOfStarGraph(int[][] e) {
        return e[0][0] == e[1][0] || e[0][0] == e[1][1] ? e[0][0] : e[0][1];
    }

    public static boolean FindIfPathExistsInGraph(int n, int[][] edges, int source, int destination) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
            graph.computeIfAbsent(v, k -> new ArrayList<>()).add(u);
        }

        Set<Integer> visited = new HashSet<>();
        return dfs(source, destination, graph, visited);
    }

    public static int FindTheTownJudge(int N, int[][] trust){
        int[] count = new int[N+1];
        for (int[] t: trust) {
            count[t[0]]--;
            count[t[1]]++;
        }
        for (int i = 1; i <= N; ++i) {
            if (count[i] == N - 1) return i;
        }
        return -1;
    }

    public static List<Integer> MinimumNumberOfVerticesToReachAllNodes(int n, List<List<Integer>> edges) {
        List<Integer> res = new ArrayList<>();
        int[] seen = new int[n];
        for (List<Integer> e: edges)
            seen[e.get(1)] = 1;
        for (int i = 0; i < n; ++i)
            if (seen[i] == 0)
                res.add(i);
        return res;
    }

    public static int MaximalNetworkRank(int n, int[][] roads) {
        int[] degree = new int[n];
        Set<String> roadSet = new HashSet<>();

        for (int[] road : roads) {
            degree[road[0]]++;
            degree[road[1]]++;
            roadSet.add(road[0] + "," + road[1]);
            roadSet.add(road[1] + "," + road[0]);
        }

        int maxRank = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                int rank = degree[i] + degree[j];
                if (roadSet.contains(i + "," + j)) {
                    rank--;
                }
                maxRank = Math.max(maxRank, rank);
            }
        }

        return maxRank;
    }
    //endregion

    //region DFS
    public static boolean FindIfPathExistsInGraph_2(int n, int[][] edges, int source, int destination){
        return FindIfPathExistsInGraph(n, edges, source, destination);
    }

    public static List<List<Integer>> AllPathsFromSourceToTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();

        path.add(0);
        dfsSearch(graph, 0, res, path);

        return res;
    }

    public static List<Integer> MinimumNumberOfVerticesToReachAllNodes_2(int n, List<List<Integer>> edges){
        return MinimumNumberOfVerticesToReachAllNodes(n, edges);
    }

    public static boolean KeysAndRooms(List<List<Integer>> rooms) {
        n = rooms.size();
        visited = new boolean[n];
        list = rooms;
        dfs(0);
        return count == n;
    }

    public static int NumberOfProvinces(int[][] mat) {
        int n = mat.length;
        boolean[] vis = new boolean[n];
        HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i <= n; ++i)
            map.put(i, new ArrayList<>());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mat[i][j] == 1 && i != j) {
                    map.get(i).add(j);
                    map.get(j).add(i);
                }
            }
        }

        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                cnt++;
                dfs(i, map, vis);
            }
        }
        return cnt;
    }
    //endregion

    //region BFS
    public static boolean FindIfPathExistsInGraph_3(int n, int[][] edges, int source, int destination) {
        return FindIfPathExistsInGraph(n, edges, source, destination);
    }

    public static List<List<Integer>> AllPathsFromSourceToTarget_2(int[][] graph){
        return AllPathsFromSourceToTarget(graph);
    }

    public static boolean KeysAndRooms_2(List<List<Integer>> rooms){
        return KeysAndRooms(rooms);
    }

    public static int NumberOfProvinces_2(int[][] mat){
        return NumberOfProvinces(mat);
    }

    public static int[] RedundantConnection(int[][] edges) {
        int[] sets = new int[edges.length + 1];

        for(int[] edge : edges) {
            int u = find(sets, edge[0]);
            int v = find(sets, edge[1]);
            if(u == v)
                return edge;
            sets[u] = v;
        }

        return new int[]{};
    }
    //endregion

    //region Strongly Connected Components
    public static int NumberOfOperationsToMakeNetworkConnected(int n, int[][] connections) {
        if (connections.length < n - 1) return -1; // To connect all nodes need at least n-1 edges
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int components = n;
        for (int[] c : connections) {
            int p1 = findParent(parent, c[0]);
            int p2 = findParent(parent, c[1]);
            if (p1 != p2) {
                parent[p1] = p2; // Union 2 component
                components--;
            }
        }
        return components - 1; // Need (components-1) cables to connect components together
    }

    public static int NumberOfProvinces_3(int[][] mat){
        return NumberOfProvinces(mat);
    }

    public static List<List<Integer>> CriticalConnectionsInANetwork(int n, List<List<Integer>> connections) {
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) graph[i] = new ArrayList<>();

        for(List<Integer> oneConnection :connections) {
            graph[oneConnection.get(0)].add(oneConnection.get(1));
            graph[oneConnection.get(1)].add(oneConnection.get(0));
        }
        int timer[] = new int[1];
        List<List<Integer>> results = new ArrayList<>();
        boolean[] visited = new boolean[n];
        int []timeStampAtThatNode = new int[n];
        criticalConnectionsUtil(graph, -1, 0, timer, visited, results, timeStampAtThatNode);
        return results;
    }

    public static List<List<Integer>> FindCriticalAndPseudoCriticalEdgesInMinimumSpanningTree(int n, int[][] edges) {
        // Step 1: Add index to edges for tracking
        int m = edges.length;
        int[][] indexedEdges = new int[m][4];
        for (int i = 0; i < m; i++) {
            indexedEdges[i][0] = edges[i][0];
            indexedEdges[i][1] = edges[i][1];
            indexedEdges[i][2] = edges[i][2];
            indexedEdges[i][3] = i;
        }
        // Sort edges by weight
        Arrays.sort(indexedEdges, Comparator.comparingInt(a -> a[2]));

        // Step 2: Compute MST weight
        int mstWeight = kruskal(n, indexedEdges, -1, -1);

        List<Integer> critical = new ArrayList<>();
        List<Integer> pseudoCritical = new ArrayList<>();

        // Step 3: Check each edge
        for (int i = 0; i < m; i++) {
            // Check if the edge is critical
            if (kruskal(n, indexedEdges, i, -1) > mstWeight) {
                critical.add(indexedEdges[i][3]);
            }
            // Check if the edge is pseudo-critical
            else if (kruskal(n, indexedEdges, -1, i) == mstWeight) {
                pseudoCritical.add(indexedEdges[i][3]);
            }
        }

        return Arrays.asList(critical, pseudoCritical);
    }

    public static int MinimumDegreeOfAConnectedTrioInAGraph(int n, int[][] edges) {
        // Step 1: Represent the graph as an adjacency matrix and degree array
        boolean[][] adj = new boolean[n + 1][n + 1];
        int[] degree = new int[n + 1];

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            adj[u][v] = true;
            adj[v][u] = true;
            degree[u]++;
            degree[v]++;
        }

        // Step 2: Identify connected trios and calculate the degree
        int minDegree = Integer.MAX_VALUE;

        for (int u = 1; u <= n; u++) {
            for (int v = u + 1; v <= n; v++) {
                if (!adj[u][v]) continue; // Skip if there's no edge between u and v

                for (int w = v + 1; w <= n; w++) {
                    if (adj[u][w] && adj[v][w]) {
                        // Trio found: {u, v, w}
                        int degreeSum = degree[u] + degree[v] + degree[w] - 6;
                        minDegree = Math.min(minDegree, degreeSum);
                    }
                }
            }
        }

        // Step 3: Return the result
        return minDegree == Integer.MAX_VALUE ? -1 : minDegree;
    }
    //endregion

    //region Backtracking
    public static List<List<Integer>> AllPathsFromSourceToTarget_3(int[][] graph){
        return AllPathsFromSourceToTarget(graph);
    }

    public static int MaximumPathQualityOfAGraph(int[] values, int[][] edges, int maxTime) {
        int n = values.length;

        // Step 1: Build the graph as an adjacency list
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1], time = edge[2];
            graph.get(u).add(new int[]{v, time});
            graph.get(v).add(new int[]{u, time});
        }

        // Step 2: Initialize variables for DFS
        boolean[] visited = new boolean[n];
        int[] maxQuality = new int[1]; // To store the maximum quality

        // Step 3: DFS function
        dfs(0, 0, 0, values, graph, maxTime, visited, maxQuality);

        return maxQuality[0];
    }
    //endregion

    //region Hashing
    public static int FindTheTownJudge_2(int N, int[][] trust){
        return FindTheTownJudge(N, trust);
    }

    public static List<Integer> MinimumNumberOfVerticesToReachAllNodes_3(int n, List<List<Integer>> edges){
        return MinimumNumberOfVerticesToReachAllNodes(n, edges);
    }

    public static int MaximalNetworkRank_2(int n, int[][] roads) {
        return MaximalNetworkRank(n, roads);
    }
    //endregion

    //region Union Find
    public static int NumberOfProvinces_4(int[][] mat){
        return NumberOfProvinces(mat);
    }

    public static int[] RedundantConnection_2(int[][] edges) {
        return RedundantConnection(edges);
    }

    public static double[] EvaluateDivision(List<List<String>> equations, double[] values, List<List<String>> queries) {
        // Step 1: Build the graph
        Map<String, Map<String, Double>> graph = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String a = equations.get(i).get(0);
            String b = equations.get(i).get(1);
            double value = values[i];

            graph.putIfAbsent(a, new HashMap<>());
            graph.putIfAbsent(b, new HashMap<>());
            graph.get(a).put(b, value);
            graph.get(b).put(a, 1.0 / value);
        }

        // Step 2: Process each query
        double[] results = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String c = queries.get(i).get(0);
            String d = queries.get(i).get(1);
            if (!graph.containsKey(c) || !graph.containsKey(d)) {
                results[i] = -1.0; // Node not in the graph
            } else {
                results[i] = dfs(c, d, new HashSet<>(), graph);
            }
        }

        return results;
    }

    public static int NumberOfOperationsToMakeNetworkConnected_2(int n, int[][] connections) {
        return NumberOfOperationsToMakeNetworkConnected(n, connections);
    }

    public static boolean IsGraphBipartite(int[][] graph) {
        int n = graph.length;
        int[] colors = new int[n]; // 0: unvisited, 1: color 1, -1: color 2

        for (int i = 0; i < n; i++) {
            if (colors[i] == 0) { // Unvisited node
                if (dfsCheck(graph, colors, i, 1)) {
                    return false;
                }
            }
        }
        return true;
    }
    //endregion

    //region Coloring
    public static boolean PossibleBiPartition(int n, int[][] dislikes) {
        // Build the adjacency list for the graph
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] dislike : dislikes) {
            graph.get(dislike[0]).add(dislike[1]);
            graph.get(dislike[1]).add(dislike[0]);
        }

        int[] colors = new int[n + 1]; // 0: unvisited, 1: group 1, -1: group 2

        for (int i = 1; i <= n; i++) {
            if (colors[i] == 0 && dfs(graph, colors, i, 1)) {
                return false;
            }
        }
        return true;
    }

    public static boolean IsGraphBipartite_2(int[][] graph){
        return IsGraphBipartite(graph);
    }

    public static boolean PossibleBiPartition_2(int n, int[][] dislikes){
        return PossibleBiPartition(n, dislikes);
    }

    public static boolean IsGraphBipartite_3(int[][] graph){
        return IsGraphBipartite(graph);
    }

    public static int[] ShortestPathWithAlternatingColors(int n, int[][] redEdges, int[][] blueEdges) {
        // Graph adjacency lists for red and blue edges
        List<Integer>[] redGraph = new ArrayList[n];
        List<Integer>[] blueGraph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            redGraph[i] = new ArrayList<>();
            blueGraph[i] = new ArrayList<>();
        }
        for (int[] edge : redEdges) {
            redGraph[edge[0]].add(edge[1]);
        }
        for (int[] edge : blueEdges) {
            blueGraph[edge[0]].add(edge[1]);
        }

        // Result array
        int[] result = new int[n];
        Arrays.fill(result, -1);

        // Queue for BFS: [node, distance, color]
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0, 0, 0}); // Start with red (0)
        queue.offer(new int[]{0, 0, 1}); // Start with blue (1)

        // Visited arrays for red and blue edges
        boolean[][] visited = new boolean[n][2]; // [node][color]
        visited[0][0] = visited[0][1] = true;

        // BFS
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int node = current[0], dist = current[1], color = current[2];

            // Update result for the current node
            if (result[node] == -1) result[node] = dist;

            // Determine the next edges to explore
            List<Integer>[] nextGraph = (color == 0) ? blueGraph : redGraph;
            int nextColor = 1 - color;

            for (int neighbor : nextGraph[node]) {
                if (!visited[neighbor][nextColor]) {
                    visited[neighbor][nextColor] = true;
                    queue.offer(new int[]{neighbor, dist + 1, nextColor});
                }
            }
        }

        return result;
    }
    //endregion

    //region Cycle Detection
    public static int[] CourseSchedule(int numCourses, int[][] prerequisites) {
        // Step 1: Build the graph and in-degree array
        List<Integer>[] graph = new ArrayList[numCourses];
        int[] inDegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] pre : prerequisites) {
            graph[pre[1]].add(pre[0]);
            inDegree[pre[0]]++;
        }

        // Step 2: Add all courses with in-degree 0 to the queue
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        // Step 3: Process the queue
        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int course = queue.poll();
            result.add(course);

            for (int neighbor : graph[course]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // Step 4: Check if all courses are processed
        if (result.size() == numCourses) {
            return result.stream().mapToInt(i -> i).toArray();
        }
        return new int[0];
    }

    public static int[] RedundantConnection_3(int[][] edges) {
        return RedundantConnection(edges);
    }

    public static boolean CourseSchedule_2(int numCourses, int[][] prerequisites) {
        // Step 1: Build the graph and in-degree array
        List<Integer>[] graph = new ArrayList[numCourses];
        int[] inDegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] pre : prerequisites) {
            graph[pre[1]].add(pre[0]);
            inDegree[pre[0]]++;
        }

        // Step 2: Add courses with in-degree 0 to the queue
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        // Step 3: Process the queue
        int processedCourses = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            processedCourses++;

            for (int neighbor : graph[course]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // Step 4: Check if all courses were processed
        return processedCourses == numCourses;
    }

    public static List<Integer> FindEventualSafeNodes(int[][] graph) {
        int n = graph.length;
        int[] visited = new int[n];  // States: 0 = unvisited, 1 = visiting, 2 = safe
        List<Integer> safeNodes = new ArrayList<>();

        // DFS function to check if a node is safe
        for (int i = 0; i < n; i++) {
            if (dfs(graph, i, visited)) {
                safeNodes.add(i);
            }
        }

        // Return the result sorted (ascending order)
        Collections.sort(safeNodes);
        return safeNodes;
    }
    //endregion

    //region Topological Sorting
    //endregion

    //region Spanning Tree
    //endregion

    //region Shortest Path
    //endregion

    //region DP Based
    //endregion

    //region Geometry
    //endregion

    //region Bit Manipulation
    //endregion

    //region Connected Components
    //endregion

    //region Eulerian Circuit
    //endregion

    //region Matrix
    //endregion

    //region Private Methods
    private static boolean dfs(int node, int destination, Map<Integer, List<Integer>> graph, Set<Integer> visited) {
        if (node == destination) {
            return true;
        }
        visited.add(node);
        for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
            if (!visited.contains(neighbor)) {
                if (dfs(neighbor, destination, graph, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void dfs(int v) {
        if(visited[v]) return;
        visited[v] = true;
        count++;
        if(count == n) return;
        for(int node : list.get(v)) {
            dfs(node);
        }
    }

    private static boolean dfs(int[][] graph, int node, int[] visited) {
        if (visited[node] == 1) {
            // If node is in visiting state, we've found a cycle
            return false;
        }
        if (visited[node] == 2) {
            // If node is already safe, return true
            return true;
        }

        // Mark node as visiting (in DFS stack)
        visited[node] = 1;

        // Explore all neighbors
        for (int neighbor : graph[node]) {
            if (!dfs(graph, neighbor, visited)) {
                return false;  // If any neighbor is unsafe, current node is unsafe
            }
        }

        // Mark node as safe after exploring all neighbors
        visited[node] = 2;
        return true;
    }

    private static void dfs(int node, HashMap<Integer, ArrayList<Integer>> map, boolean[] vis) {
        vis[node] = true;
        for (Integer neigh : map.get(node)) {
            if (!vis[neigh])
                dfs(neigh, map, vis);
        }
    }

    private static void dfs(int node, int currentQuality, int currentTime, int[] values,
                     Map<Integer, List<int[]>> graph, int maxTime,
                     boolean[] visited, int[] maxQuality) {
        // Visit the node and update quality
        boolean isFirstVisit = !visited[node];
        if (isFirstVisit) {
            currentQuality += values[node];
        }
        visited[node] = true;

        // If we return to node 0, update the max quality
        if (node == 0) {
            maxQuality[0] = Math.max(maxQuality[0], currentQuality);
        }

        // Explore neighbors
        for (int[] neighbor : graph.get(node)) {
            int nextNode = neighbor[0];
            int travelTime = neighbor[1];
            if (currentTime + travelTime <= maxTime) {
                dfs(nextNode, currentQuality, currentTime + travelTime, values, graph, maxTime, visited, maxQuality);
            }
        }

        // Backtrack
        visited[node] = !isFirstVisit && visited[node];
    }

    private static double dfs(String current, String target, Set<String> visited, Map<String, Map<String, Double>> graph) {
        // Base case: If the target is found
        if (current.equals(target)) return 1.0;

        // Mark the current node as visited
        visited.add(current);

        // Explore neighbors
        for (Map.Entry<String, Double> neighbor : graph.get(current).entrySet()) {
            String nextNode = neighbor.getKey();
            double weight = neighbor.getValue();

            if (!visited.contains(nextNode)) {
                double result = dfs(nextNode, target, visited, graph);
                if (result != -1.0) {
                    return result * weight;
                }
            }
        }

        // Backtrack and return -1.0 if no path is found
        return -1.0;
    }

    private static boolean dfs(List<List<Integer>> graph, int[] colors, int node, int color) {
        colors[node] = color;
        for (int neighbor : graph.get(node)) {
            if (colors[neighbor] == 0) {
                if (dfs(graph, colors, neighbor, -color)) {
                    return true;
                }
            } else if (colors[neighbor] == color) {
                return true;
            }
        }
        return false;
    }

    private static boolean dfsCheck(int[][] graph, int[] colors, int node, int color) {
        colors[node] = color;
        for (int neighbor : graph[node]) {
            if (colors[neighbor] == 0) { // If unvisited, color it with opposite color
                if (dfsCheck(graph, colors, neighbor, -color)) {
                    return true;
                }
            } else if (colors[neighbor] == color) { // If neighbor has the same color, not bipartite
                return true;
            }
        }
        return false;
    }

    private static void dfsSearch(int[][] graph, int node, List<List<Integer>> res, List<Integer> path) {
        if (node == graph.length - 1) {
            res.add(new ArrayList<Integer>(path));
            return;
        }

        for (int nextNode : graph[node]) {
            path.add(nextNode);
            dfsSearch(graph, nextNode, res, path);
            path.remove(path.size() - 1);
        }
    }

    private static int find(int[] sets, int v) {
        return sets[v] == 0 ? v : find(sets, sets[v]);
    }

    private static int findParent(int[] parent, int i) {
        while (i != parent[i]) i = parent[i];
        return i; // Without Path Compression
    }

    public static void criticalConnectionsUtil(List<Integer>[] graph, int parent, int node, int[] timer, boolean[] visited, List<List<Integer>> results, int []timeStampAtThatNode) {
        visited[node] = true;
        timeStampAtThatNode[node] = timer[0]++;
        int currentTimeStamp = timeStampAtThatNode[node];

        for(int oneNeighbour : graph[node]) {
            if(oneNeighbour == parent) continue;
            if(!visited[oneNeighbour]) criticalConnectionsUtil(graph, node, oneNeighbour, timer, visited, results, timeStampAtThatNode);
            timeStampAtThatNode[node] = Math.min(timeStampAtThatNode[node], timeStampAtThatNode[oneNeighbour]);
            if(currentTimeStamp < timeStampAtThatNode[oneNeighbour]) results.add(Arrays.asList(node, oneNeighbour));
        }
    }

    private static int kruskal(int n, int[][] edges, int excludeEdge, int includeEdge) {
        UnionFind uf = new UnionFind(n);
        int mstWeight = 0;
        int edgesUsed = 0;

        // Include the mandatory edge if specified
        if (includeEdge != -1) {
            int[] edge = edges[includeEdge];
            if (uf.union(edge[0], edge[1])) {
                mstWeight += edge[2];
                edgesUsed++;
            }
        }

        // Process all edges except the excluded one
        for (int i = 0; i < edges.length; i++) {
            if (i == excludeEdge) continue;
            int[] edge = edges[i];
            if (uf.union(edge[0], edge[1])) {
                mstWeight += edge[2];
                edgesUsed++;
            }
        }

        // If we used exactly n - 1 edges, it's a valid MST
        return edgesUsed == n - 1 ? mstWeight : Integer.MAX_VALUE;
    }
    //endregion
}
