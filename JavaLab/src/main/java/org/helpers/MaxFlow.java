package org.helpers;

import java.util.*;

public class MaxFlow {
    public static class Edge {
        public int to, rev;
        public long capacity;

        public Edge(int to, int rev, long capacity) {
            this.to = to;
            this.rev = rev;
            this.capacity = capacity;
        }
    }

    public static class FlowNetwork {
        public int n;
        public List<Edge>[] graph;

        @SuppressWarnings("unchecked")
        public FlowNetwork(int n) {
            this.n = n;
            graph = new ArrayList[n];
            for (int i = 0; i < n; i++) {
                graph[i] = new ArrayList<>();
            }
        }

        public void addEdge(int from, int to, long capacity) {
            Edge forward = new Edge(to, graph[to].size(), capacity);
            Edge backward = new Edge(from, graph[from].size(), 0);
            graph[from].add(forward);
            graph[to].add(backward);
        }
    }

    public static class Dinic {
        FlowNetwork network;
        int[] level;
        int[] ptr;
        int source, sink;

        public Dinic(FlowNetwork network, int source, int sink) {
            this.network = network;
            this.source = source;
            this.sink = sink;
            this.level = new int[network.n];
            this.ptr = new int[network.n];
        }

        public boolean bfs() {
            Arrays.fill(level, -1);
            Queue<Integer> queue = new LinkedList<>();
            queue.add(source);
            level[source] = 0;

            while (!queue.isEmpty()) {
                int node = queue.poll();
                for (Edge edge : network.graph[node]) {
                    if (edge.capacity > 0 && level[edge.to] == -1) {
                        level[edge.to] = level[node] + 1;
                        queue.add(edge.to);
                        if (edge.to == sink) {
                            return true;
                        }
                    }
                }
            }
            return level[sink] != -1;
        }

        public long dfs(int node, long pushed) {
            if (pushed == 0) return 0;
            if (node == sink) return pushed;

            for (; ptr[node] < network.graph[node].size(); ptr[node]++) {
                Edge edge = network.graph[node].get(ptr[node]);
                if (level[edge.to] == level[node] + 1 && edge.capacity > 0) {
                    long flow = dfs(edge.to, Math.min(pushed, edge.capacity));
                    if (flow > 0) {
                        edge.capacity -= flow;
                        network.graph[edge.to].get(edge.rev).capacity += flow;
                        return flow;
                    }
                }
            }
            return 0;
        }

        public long maxFlow() {
            long flow = 0;
            while (bfs()) {
                Arrays.fill(ptr, 0);
                while (true) {
                    long pushed = dfs(source, Long.MAX_VALUE);
                    if (pushed == 0) break;
                    flow += pushed;
                }
            }
            return flow;
        }
    }
}
