package org.tgp;

import java.util.*;

public class Level_C6 {
    public static List<Integer> MedianInADataStream(List<Integer> arr, int n) {
        List<Integer> result = new ArrayList<>();
        PriorityQueue<Integer> lowerHalf = new PriorityQueue<>(Collections.reverseOrder()); // Max-Heap
        PriorityQueue<Integer> upperHalf = new PriorityQueue<>(); // Min-Heap

        for (int num : arr) {
            // Insert into appropriate heap
            if (lowerHalf.isEmpty() || num <= lowerHalf.peek()) {
                lowerHalf.add(num);
            } else {
                upperHalf.add(num);
            }

            // Balance the heaps
            if (lowerHalf.size() > upperHalf.size() + 1) {
                upperHalf.add(lowerHalf.poll());
            } else if (upperHalf.size() > lowerHalf.size()) {
                lowerHalf.add(upperHalf.poll());
            }

            // Calculate median
            if (lowerHalf.size() == upperHalf.size()) {
                result.add((lowerHalf.peek() + upperHalf.peek()) / 2);
            } else {
                result.add(lowerHalf.peek());
            }
        }

        return result;
    }

    public static List<List<Integer>> RotateImage(List<List<Integer>> matrix) {
        int n = matrix.size();

        // Step 1: Transpose the matrix
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // Swap elements (i, j) and (j, i)
                int temp = matrix.get(i).get(j);
                matrix.get(i).set(j, matrix.get(j).get(i));
                matrix.get(j).set(i, temp);
            }
        }

        // Step 2: Reverse each row
        for (int i = 0; i < n; i++) {
            Collections.reverse(matrix.get(i));
        }
        return matrix;
    }

    public int[][] ReconstructQueue(int[][] people) {
        // Step 1: Sort by height (descending), and by k (ascending)
        Arrays.sort(people, (a, b) -> {
            if (a[0] != b[0]) {
                return b[0] - a[0]; // Descending order of height
            } else {
                return a[1] - b[1]; // Ascending order of k
            }
        });

        // Step 2: Insert into queue based on k value
        List<int[]> queue = new LinkedList<>();
        for (int[] person : people) {
            queue.add(person[1], person); // Insert at index k
        }

        return queue.toArray(new int[queue.size()][]);

    }
}
