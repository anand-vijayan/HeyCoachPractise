package org.heycoach;

import java.util.*;
import static org.modules.AdvancedDataStructure.*;

public class Main {
    public static void main(String[] args)  {

        // Example 1
        int N = 6, E = 7;
        int[][] edges = new int[][] {{0, 1},{0, 2},{1, 2},{1, 3},{2, 4},{3, 4},{3, 5}};
        int S = 0, D = 5;

        System.out.println(shortestPath(N, edges, S, D));  // Output: 3

        // Example 2
        N = 5; E = 5;
        edges = new int[][] {{0, 1}, {1, 2}, {1, 3}, {2, 3}, {3, 4}};
        S = 0; D = 4;

        System.out.println(shortestPath(N, edges, S, D));  // Output: 3
    }
}

