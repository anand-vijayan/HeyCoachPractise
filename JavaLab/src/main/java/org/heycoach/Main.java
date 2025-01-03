package org.heycoach;

import java.util.*;
import static org.modules.GraphAlgorithms.*;

public class Main {
    public static void main(String[] args)  {
        int n = 4;
        int m = 5;
        int[][] edges = new int[m][];
        edges[0] = new int[] {1,2,1};
        edges[1] = new int[] {1,3,2};
        edges[2] = new int[] {1,4,3};
        edges[3] = new int[] {2,3,5};
        edges[4] = new int[] {3,4,4};

        int[][] result = MinimumSpanningTree(n, m, edges);
        System.out.println(result[0][0]);
    }
}

