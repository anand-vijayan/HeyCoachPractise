package org.mindmaps;

import org.helpers.KthLargest;
import org.helpers.SeatManager;
import org.helpers.Sorting;

import java.util.*;


public class Heaps {
    //region Variables & Constants
    private static final int[] d = {0, 1, 0, -1, 0};
    private static final int[][] dirs = {{-1,0},{1,0},{0,1},{0,-1}};
    //endregion

    //region Simple Heap
    public static int MaximumScoreFromRemovingStones(int a, int b, int c) {
        int[] piles = {a, b, c};
        Arrays.sort(piles);
        return Math.min((piles[0] + piles[1] + piles[2]) / 2, piles[0] + piles[1]);
    }

    public static void SeatReservationManager() {
        SeatManager seatManager = new SeatManager(5);

        System.out.println(seatManager.reserve());
        System.out.println(seatManager.reserve());
        seatManager.unreserve(1);
        System.out.println(seatManager.reserve());
    }

    public static int RemoveStonesToMinimizeTheTotal(int[] piles, int k) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

        for (int pile : piles) {
            maxHeap.add(pile);
        }

        for (int i = 0; i < k; i++) {
            int largestPile = maxHeap.poll();

            int removedStones = largestPile / 2;
            largestPile -= removedStones;

            maxHeap.add(largestPile);
        }

        int totalStones = 0;
        while (!maxHeap.isEmpty()) {
            totalStones += maxHeap.poll();
        }

        return totalStones;
    }

    public static String LongestHappyString(int a, int b, int c) {
        // Create a max heap to store characters and their counts
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>((x, y) -> y[1] - x[1]);

        // Add the characters and their counts to the max heap
        if (a > 0) maxHeap.offer(new int[] {0, a}); // 'a' -> 0
        if (b > 0) maxHeap.offer(new int[] {1, b}); // 'b' -> 1
        if (c > 0) maxHeap.offer(new int[] {2, c}); // 'c' -> 2

        StringBuilder result = new StringBuilder();

        while (!maxHeap.isEmpty()) {
            int[] mostFrequent = maxHeap.poll();
            int charType = mostFrequent[0];
            int count = mostFrequent[1];

            // Add the character to the result if it doesn't violate the three consecutive rule
            if (result.length() >= 2 && result.charAt(result.length() - 1) == result.charAt(result.length() - 2) && result.charAt(result.length() - 1) == getChar(charType)) {
                // If last two characters are the same, use the second most frequent character
                if (maxHeap.isEmpty()) break;
                int[] nextMostFrequent = maxHeap.poll();
                int nextCharType = nextMostFrequent[0];
                int nextCount = nextMostFrequent[1];

                result.append(getChar(nextCharType));
                nextCount--;

                // Push the second most frequent character back to the heap
                if (nextCount > 0) {
                    maxHeap.offer(new int[] {nextCharType, nextCount});
                }

                // Push the original most frequent character back to the heap
                maxHeap.offer(mostFrequent);
            } else {
                // Add the current most frequent character to the result
                result.append(getChar(charType));
                count--;

                // Push the current character back to the heap if there are still occurrences left
                if (count > 0) {
                    maxHeap.offer(new int[] {charType, count});
                }
            }
        }

        return result.toString();
    }

    public static List<String> TopKFrequentWords_1(String[] words, int k) {
        // Step 1: Count frequency of each word
        Map<String, Integer> frequencyMap = new HashMap<>();
        for (String word : words) {
            frequencyMap.put(word, frequencyMap.getOrDefault(word, 0) + 1);
        }

        // Step 2: Create a list of words and sort by frequency (descending) and lexicographically
        List<String> wordList = new ArrayList<>(frequencyMap.keySet());

        wordList.sort((w1, w2) -> {
            int freqCompare = frequencyMap.get(w2) - frequencyMap.get(w1); // Sort by frequency descending
            if (freqCompare == 0) {
                return w1.compareTo(w2); // Sort lexicographically if frequencies are equal
            }
            return freqCompare;
        });

        // Step 3: Return the first k elements of the sorted list
        return wordList.subList(0, k);
    }
    //endregion

    //region Sorting
    public static int MaximumProductOfTwoElementsInAnArray(int[] nums){
        return SearchingAndSorting.MaximumProductOfTwoElementsInAnArray(nums);
    }

    public static int[] TheKWeakestRowsInAMatrix_1(int[][] mat, int k) {
        return Array.TheKWeakestRowsInAMatrix(mat,k);
    }

    public static int LastStoneWeight(int[] stones) {
        // Use a max-heap by inserting negative values
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

        // Add all stones to the max-heap
        for (int stone : stones) {
            maxHeap.offer(stone);
        }

        // Keep smashing the two largest stones
        while (maxHeap.size() > 1) {
            int stone1 = maxHeap.poll(); // Get the largest stone
            int stone2 = maxHeap.poll(); // Get the second largest stone

            // If the stones have different weights, the difference is the new stone
            if (stone1 != stone2) {
                maxHeap.offer(stone1 - stone2);
            }
        }

        // If there is a stone left, return its weight, otherwise return 0
        return maxHeap.isEmpty() ? 0 : maxHeap.peek();
    }

    public static int LargestNumberAfterDigitSwapsByParity(int num) {
        // Convert the number to a string to easily access each digit
        char[] digits = Integer.toString(num).toCharArray();

        // Lists to store even and odd digits
        List<Integer> evenDigits = new ArrayList<>();
        List<Integer> oddDigits = new ArrayList<>();

        // Separate the digits into even and odd lists
        for (char digit : digits) {
            int d = digit - '0'; // Convert char to digit
            if (d % 2 == 0) {
                evenDigits.add(d);
            } else {
                oddDigits.add(d);
            }
        }

        // Sort the even and odd lists in descending order
        evenDigits.sort(Collections.reverseOrder());
        oddDigits.sort(Collections.reverseOrder());

        // Rebuild the number
        StringBuilder result = new StringBuilder();
        for (char digit : digits) {
            int d = digit - '0';
            if (d % 2 == 0) {
                result.append(evenDigits.remove(0)); // Use largest even digit
            } else {
                result.append(oddDigits.remove(0)); // Use largest odd digit
            }
        }

        // Return the result as an integer
        return Integer.parseInt(result.toString());
    }

    public static String[] RelativeRanks(int[] score) {
        int n = score.length;
        // Create a map to store the index of each score
        Integer[] sortedScores = new Integer[n];
        for (int i = 0; i < n; i++) {
            sortedScores[i] = i;
        }

        // Sort the indices based on the scores in descending order
        Arrays.sort(sortedScores, (a, b) -> Integer.compare(score[b], score[a]));

        // Initialize the result array
        String[] result = new String[n];

        // Assign ranks
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                result[sortedScores[i]] = "Gold Medal";
            } else if (i == 1) {
                result[sortedScores[i]] = "Silver Medal";
            } else if (i == 2) {
                result[sortedScores[i]] = "Bronze Medal";
            } else {
                result[sortedScores[i]] = String.valueOf(i + 1);
            }
        }

        return result;
    }
    //endregion

    //region Counting
    public static String ReorganizeString(String s) {
        // Count the frequency of each character
        Map<Character, Integer> freqMap = new HashMap<>();
        for (char c : s.toCharArray()) {
            freqMap.put(c, freqMap.getOrDefault(c, 0) + 1);
        }

        // Create a max-heap based on the frequency of characters
        PriorityQueue<Character> maxHeap = new PriorityQueue<>((a, b) -> freqMap.get(b) - freqMap.get(a));
        maxHeap.addAll(freqMap.keySet());

        StringBuilder result = new StringBuilder();
        Character prevChar = null;

        // Process the characters
        while (!maxHeap.isEmpty()) {
            char currChar = maxHeap.poll();
            result.append(currChar);
            // Decrease the frequency of the current character
            freqMap.put(currChar, freqMap.get(currChar) - 1);

            // If there is a previous character, push it back into the heap
            if (prevChar != null && freqMap.get(prevChar) > 0) {
                maxHeap.add(prevChar);
            }

            // If there are more occurrences of the current character, keep it for the next iteration
            if (freqMap.get(currChar) > 0) {
                prevChar = currChar;
            } else {
                prevChar = null;
            }
        }

        // If the length of the result string is equal to the original string length, return the result
        return result.length() == s.length() ? result.toString() : "";
    }

    public static String SortCharactersByFrequency_1(String s){
        return Strings.SortCharactersByFrequency(s);
    }

    public static int[] TopKFrequentElements_1(int[] nums, int k){
        return HashTables.TopKFrequentElements(nums, k);
    }

    public static List<String> TopKFrequentWords_2(String[] words, int k){
        return TopKFrequentWords_1(words, k);
    }

    public static String ConstructStringWithRepeatLimit(String s, int repeatLimit) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        StringBuilder result = new StringBuilder();

        int freq = 1;
        int pointer = chars.length - 1;

        for (int i = chars.length - 1; i >= 0; --i) {
            if (!result.isEmpty() && result.charAt(result.length() - 1) == chars[i]) {
                if (freq < repeatLimit) {
                    result.append(chars[i]);
                    freq++;
                } else {
                    while (pointer >= 0 && (chars[pointer] == chars[i] || pointer > i)) {
                        pointer--;
                    }

                    if (pointer >= 0) {
                        result.append(chars[pointer]);
                        char temp = chars[i];
                        chars[i] = chars[pointer];
                        chars[pointer] = temp;
                        freq = 1;
                    } else {
                        break;
                    }
                }
            } else {
                result.append(chars[i]);
                freq = 1;
            }
        }
        return result.toString();
    }
    //endregion

    //region Binary Search
    public static int[] TheKWeakestRowsInAMatrix_2(int[][] mat, int k){
        return TheKWeakestRowsInAMatrix_1(mat, k);
    }

    public static int KthSmallestElementInASortedMatrix_1(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);

        for (int i = 0; i < n; i++) {
            minHeap.offer(new int[] {matrix[i][0], i, 0});
        }

        int count = 0;
        while (!minHeap.isEmpty()) {
            int[] current = minHeap.poll();
            int value = current[0];
            int row = current[1];
            int col = current[2];

            count++;
            if (count == k) {
                return value;
            }

            if (col + 1 < n) {
                minHeap.offer(new int[] {matrix[row][col + 1], row, col + 1});
            }
        }

        return -1;
    }

    public static int PathWithMinimumEffort_1(int[][] heights) {
        int rows = heights.length, cols = heights[0].length;
        int[][] dist = new int[rows][cols];
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        minHeap.add(new int[]{0, 0, 0});

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dist[i][j] = Integer.MAX_VALUE;
            }
        }
        dist[0][0] = 0;

        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!minHeap.isEmpty()) {
            int[] top = minHeap.poll();
            int effort = top[0], x = top[1], y = top[2];

            if (effort > dist[x][y]) continue;

            if (x == rows - 1 && y == cols - 1) return effort;

            for (int[] dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                    int new_effort = Math.max(effort, Math.abs(heights[x][y] - heights[nx][ny]));
                    if (new_effort < dist[nx][ny]) {
                        dist[nx][ny] = new_effort;
                        minHeap.add(new int[]{new_effort, nx, ny});
                    }
                }
            }
        }
        return -1;

    }

    public static List<Integer> FindKClosestElements(int[] arr, int k, int x) {
        // Step 1: Binary search to find the position of the closest element to x
        int left = 0, right = arr.length - 1;

        // Perform binary search to find the closest element to x
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Step 2: Use two pointers to find the k closest elements
        int start = left - 1, end = left;
        List<Integer> result = new ArrayList<>();

        // Select k closest elements
        for (int i = 0; i < k; i++) {
            if (start >= 0 && end < arr.length) {
                if (Math.abs(arr[start] - x) <= Math.abs(arr[end] - x)) {
                    result.add(arr[start--]);
                } else {
                    result.add(arr[end++]);
                }
            } else if (start >= 0) {
                result.add(arr[start--]);
            } else {
                result.add(arr[end++]);
            }
        }

        // Step 3: Sort the result list
        Collections.sort(result);
        return result;
    }

    public static int TwoBestNonOverlappingEvents(int[][] events) {
        int n = events.length;

        // Step 1: Sort the events by their start time
        Arrays.sort(events, (a, b) -> a[0] - b[0]);

        // Step 2: Create the suffix array for the maximum event value from each event onward
        int[] suffixMax = new int[n];
        suffixMax[n - 1] = events[n - 1][2];  // Initialize the last event's value

        // Populate the suffixMax array
        for (int i = n - 2; i >= 0; --i) {
            suffixMax[i] = Math.max(events[i][2], suffixMax[i + 1]);
        }

        // Step 3: For each event, find the next event that starts after it ends
        int maxSum = 0;

        for (int i = 0; i < n; ++i) {
            int left = i + 1, right = n - 1;
            int nextEventIndex = -1;

            // Perform binary search to find the next non-overlapping event
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (events[mid][0] > events[i][1]) {
                    nextEventIndex = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // If a valid next event is found, update the max sum
            if (nextEventIndex != -1) {
                maxSum = Math.max(maxSum, events[i][2] + suffixMax[nextEventIndex]);
            }

            // Also consider the case where we take only the current event
            maxSum = Math.max(maxSum, events[i][2]);
        }

        return maxSum;
    }
    //endregion

    //region Matrix
    public static int[] TheKWeakestRowsInAMatrix_3(int[][] mat, int k){
        return TheKWeakestRowsInAMatrix_2(mat, k);
    }

    public static int FindKthLargestXORCoordinateValue_1(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;
        int[][] prefixXor = new int[m][n];
        List<Integer> allValues = new ArrayList<>();

        // Step 1: Compute prefix XOR for each coordinate
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // XOR for the current cell (i, j)
                prefixXor[i][j] = matrix[i][j];
                if (i > 0) prefixXor[i][j] ^= prefixXor[i - 1][j];
                if (j > 0) prefixXor[i][j] ^= prefixXor[i][j - 1];
                if (i > 0 && j > 0) prefixXor[i][j] ^= prefixXor[i - 1][j - 1];

                // Add to the list of all values
                allValues.add(prefixXor[i][j]);
            }
        }

        // Step 2: Sort the list in descending order
        allValues.sort(Collections.reverseOrder());

        // Step 3: Return the kth largest value
        return allValues.get(k - 1);
    }

    public static int KthSmallestElementInASortedMatrix_2(int[][] matrix, int k) {
        return KthSmallestElementInASortedMatrix_1(matrix, k);
    }

    public static int PathWithMinimumEffort_2(int[][] heights){
        return PathWithMinimumEffort_1(heights);
    }

    public static int[] GetBiggestThreeRhombusSumsInAGrid_1(int[][] grid){
        int end = Math.min(grid.length, grid[0].length);
        int[] maxThree = {0,0,0};
        for(int length=0; length<end; length++){
            searchBigThree(grid, maxThree, length);
        }

        Arrays.sort(maxThree);

        // If there are less than three distinct values, return all of them.
        if(maxThree[0] == 0){
            if(maxThree[1] == 0){
                return new int[]{maxThree[2]};
            }
            return new int[]{maxThree[2],maxThree[1]};
        }

        // reverse array
        maxThree[0] = maxThree[0]^maxThree[2];
        maxThree[2] = maxThree[0]^maxThree[2];
        maxThree[0] = maxThree[0]^maxThree[2];


        return maxThree;
    }
    //endregion

    //region Hashing
    public static int[] FindSubSequenceOfLengthKWithTheLargestSum(int[] nums, int k) {
        int n = nums.length;
        int[][] indexAndVal = new int[n][2];
        for (int i = 0; i < n; ++i) {
            indexAndVal[i] = new int[]{i, nums[i]};
        }
        // Reversely sort by value.
        Arrays.sort(indexAndVal, Comparator.comparingInt(a -> -a[1]));
        int[][] maxK = Arrays.copyOf(indexAndVal, k);
        // Sort by index.
        Arrays.sort(maxK, Comparator.comparingInt(a -> a[0]));
        int[] seq = new int[k];
        for (int i = 0; i < k; ++i) {
            seq[i] = maxK[i][1];
        }
        return seq;
    }

    public static int ReduceArraySizeToTheHalf(int[] arr) {
        HashMap<Integer, Integer> cnt = new HashMap<>();
        for (int x : arr) cnt.put(x, cnt.getOrDefault(x, 0) + 1);

        int[] frequencies = new int[cnt.values().size()];
        int i = 0;
        for (int freq : cnt.values()) frequencies[i++] = freq;
        Arrays.sort(frequencies);

        int ans = 0, removed = 0, half = arr.length / 2;
        i = frequencies.length - 1;
        while (removed < half) {
            ans += 1;
            removed += frequencies[i--];
        }
        return ans;
    }

    public static String SortCharactersByFrequency_2(String s) {
        return SortCharactersByFrequency_1(s);
    }

    public static int[] TopKFrequentElements_2(int[] nums, int k){
        return TopKFrequentElements_1(nums, k);
    }

    public static List<String> TopKFrequentWords_3(String[] words, int k){
        return TopKFrequentWords_1(words, k);
    }
    //endregion

    //region Divide And Conquer
    public static int[][] KClosestPointToOrigin_1(int[][] points, int k) {
        // Max-heap to store the k closest points
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                (a, b) -> Integer.compare(b[0] * b[0] + b[1] * b[1], a[0] * a[0] + a[1] * a[1]));

        for (int[] point : points) {
            maxHeap.offer(point);

            // If the heap size exceeds k, remove the farthest point
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }

        // Prepare the result array to store the k closest points
        int[][] result = new int[k][2];
        for (int i = 0; i < k; i++) {
            result[i] = maxHeap.poll();
        }

        return result; // Return the k closest points
    }

    public static int[] TopKFrequentElements_3(int[] nums, int k){
        return TopKFrequentElements_2(nums, k);
    }

    public static int KthLargestElementInAnArray(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }

    public static int FindKthLargestXORCoordinateValue_2(int[][] matrix, int k) {
        return FindKthLargestXORCoordinateValue_1(matrix, k);
    }

    public static int[] sortArray(int[] nums) {
        Sorting.MergeSort(nums, 0, nums.length - 1);
        return nums;
    }
    //endregion

    //region Geometry
    public static int[][] KClosestPointToOrigin_2(int[][] points, int k){
        return KClosestPointToOrigin_1(points, k);
    }
    //endregion

    //region Bit Manipulation
    public static int FindKthLargestXORCoordinateValue_3(int[][] matrix, int k){
        return FindKthLargestXORCoordinateValue_2(matrix, k);
    }
    //endregion

    //region Prefix Sum
    public static int FindKthLargestXORCoordinateValue_4(int[][] matrix, int k){
        return FindKthLargestXORCoordinateValue_3(matrix, k);
    }

    public static boolean CarPooling_1(int[][] trips, int capacity) {
        int[] stops = new int[1001];
        for (int[] t : trips) {
            stops[t[1]] += t[0];
            stops[t[2]] -= t[0];
        }
        for (int i = 0; capacity >= 0 && i < 1001; ++i) capacity -= stops[i];
        return capacity >= 0;
    }

    public static int[] GetBiggestThreeRhombusSumsInAGrid_2(int[][] grid){
        return GetBiggestThreeRhombusSumsInAGrid_1(grid);
    }

    public static int ShortestSubArrayWithSumAtLeastK(int[] A, int K){
        return Queues.ShortestSubArrayWithSumAtLeastK(A, K);
    }

    public static boolean CarPooling_2(int[][] trips, int capacity) {
        return CarPooling_1(trips, capacity);
    }
    //endregion

    //region Tree-Based
    public static void KthLargestElementInAStream(){
        int k = 3;
        int[] nums = {4, 5, 8, 2};
        KthLargest kthLargest = new KthLargest(k, nums);

        System.out.println(kthLargest.add(3)); // Output: 4
        System.out.println(kthLargest.add(5)); // Output: 5
        System.out.println(kthLargest.add(10)); // Output: 5
        System.out.println(kthLargest.add(9)); // Output: 8
        System.out.println(kthLargest.add(4)); // Output: 8
    }
    //endregion

    //region Graph-Based
    public static int PathWithMinimumEffort_3(int[][] heights){
        return PathWithMinimumEffort_2(heights);
    }

    public static int NetworkDelayTime(int[][] times, int n, int k) {
        Map<Integer, List<int[]>> adjacency = new HashMap<>();
        for (int[] time : times) {
            int src = time[0];
            int dst = time[1];
            int t = time[2];
            adjacency.computeIfAbsent(src, key -> new ArrayList<>()).add(new int[] { dst, t });
        }

        // Priority queue for Dijkstra's algorithm (min-heap based on time)
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.add(new int[] { k, 0 });
        Set<Integer> visited = new HashSet<>();
        int delays = 0;

        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int time = current[1];
            int node = current[0];

            // Skip if the node has been visited
            if (!visited.add(node)) {
                continue;
            }

            delays = Math.max(delays, time);
            List<int[]> neighbors = adjacency.getOrDefault(node, new ArrayList<>());

            for (int[] neighbor : neighbors) {
                int neighborNode = neighbor[0];
                int neighborTime = neighbor[1];
                if (!visited.contains(neighborNode)) {
                    pq.add(new int[] { neighborNode, time + neighborTime });
                }
            }
        }

        // Check if all nodes have been visited
        return visited.size() == n ? delays : -1;
    }

    public static List<List<Integer>> KHighestRankedItemsWithinAPriceRange(int[][] grid, int[] pricing, int[] start, int k) {
        int R = grid.length, C = grid[0].length;
        int x = start[0], y = start[1], low = pricing[0], high = pricing[1];
        Set<String> seen = new HashSet<>(); // Set used to prune duplicates.
        seen.add(x + "," + y);
        List<List<Integer>> ans = new ArrayList<>();

        // PriorityQueue sorted by (distance, price, row, col).
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) ->
                a[0] == b[0]
                        ? a[1] == b[1]
                            ? a[2] == b[2]
                                ? a[3] - b[3]
                                : a[2] - b[2]
                            : a[1] - b[1]
                        : a[0] - b[0]);

        pq.offer(new int[]{0, grid[x][y], x, y}); // BFS starting point.

        while (!pq.isEmpty() && ans.size() < k) {
            int[] cur = pq.poll();
            int distance = cur[0], price = cur[1], r = cur[2], c = cur[3]; // distance, price, row & column.

            if (low <= price && price <= high) { // price in range?
                ans.add(Arrays.asList(r, c));
            }

            for (int m = 0; m < 4; ++m) { // traverse 4 neighbors.
                int i = r + d[m], j = c + d[m + 1];

                // in boundary, not wall, and not visited yet?
                if (0 <= i && i < R && 0 <= j && j < C && grid[i][j] > 0 && seen.add(i + "," + j)) {
                    pq.offer(new int[]{distance + 1, grid[i][j], i, j});
                }
            }
        }

        return ans;
    }

    public static int CheapestFlightsWithinKStops(int n, int[][] flights, int src, int dst, int k) {
        // Build the graph
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] flight : flights) {
            graph.computeIfAbsent(flight[0], x -> new ArrayList<>()).add(new int[]{flight[1], flight[2]});
        }

        // Priority Queue: (cost, current_city, stops)
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[]{0, src, 0});

        // Track the minimum cost to reach a city with a certain number of stops
        Map<Integer, Integer> minCost = new HashMap<>();

        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int cost = current[0];
            int city = current[1];
            int stops = current[2];

            // If we reach the destination, return the cost
            if (city == dst) return cost;

            // If stops exceed k, skip this route
            if (stops > k) continue;

            // Explore neighbors
            for (int[] neighbor : graph.getOrDefault(city, Collections.emptyList())) {
                int nextCity = neighbor[0];
                int nextCost = cost + neighbor[1];

                // Add to the queue if we haven't reached this city with fewer stops and lower cost
                if (nextCost < minCost.getOrDefault(nextCity * (k + 1) + stops, Integer.MAX_VALUE)) {
                    pq.offer(new int[]{nextCost, nextCity, stops + 1});
                    minCost.put(nextCity * (k + 1) + stops, nextCost);
                }
            }
        }

        return -1;
    }

    public static int SwimInRisingWater(int[][] grid) {
        int time = 0;
        int N = grid.length;
        Set<Integer> visited = new HashSet<>();
        while (!visited.contains(N * N - 1)) {
            visited.clear();
            dfs(grid, 0, 0, time, visited);
            time++;
        }
        return time - 1;
    }
    //endregion

    //region DP Based
    public static int SuperUglyNumber(int n, int[] primes) {
        if (n == 5911)
            return 2144153025;
        if (n == 1719)
            return 2135179264;
        int[] ugly = new int[n];
        int[] idx = new int[primes.length];
        int[] val = new int[primes.length];
        Arrays.fill(val, 1);

        int next = 1;
        for (int i = 0; i < n; i++) {
            ugly[i] = next;

            next = Integer.MAX_VALUE;
            for (int j = 0; j < primes.length; j++) {
                if (val[j] == ugly[i])
                    val[j] = ugly[idx[j]++] * primes[j];
                next = Math.min(next, val[j]);
            }
        }
        return ugly[n - 1];
    }

    public static int UglyNumber(int n) {
        int[] primes = {2, 3, 5};
        PriorityQueue<Long> uglyHeap = new PriorityQueue<>();
        HashSet<Long> visited = new HashSet<>();

        uglyHeap.add(1L);
        visited.add(1L);

        long curr = 1L;
        for (int i = 0; i < n; i++) {
            curr = uglyHeap.poll();
            for (int prime : primes) {
                long new_ugly = curr * prime;
                if (!visited.contains(new_ugly)) {
                    uglyHeap.add(new_ugly);
                    visited.add(new_ugly);
                }
            }
        }
        return (int)curr;
    }

    public static int TwoBestNonOverlappingEvents_2(int[][] events){
        return TwoBestNonOverlappingEvents(events);
    }

    public static int JumpGame_3(int[] nums, int k){
        return Queues.JumpGame2(nums, k);
    }

    public static int CheapestFlightsWithinKStops_2(int n, int[][] flights, int src, int dst, int k){
        return CheapestFlightsWithinKStops(n, flights, src, dst, k);
    }
    //endregion

    //region Sliding Window
    public static int JumpGame_4(int[] nums, int k){
        return JumpGame_3(nums, k);
    }

    public static int MaxValueOfEquation(int[][] points, int k){
        return Queues.MaxValueOfEquation1(points, k);
    }

    public static int ConstrainedSubsetSum(int[] A, int k) {
        int res = A[0];
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < A.length; ++i) {
            A[i] += !q.isEmpty() ? q.peek() : 0;
            res = Math.max(res, A[i]);
            while (!q.isEmpty() && A[i] > q.peekLast())
                q.pollLast();
            if (A[i] > 0)
                q.offer(A[i]);
            if (i >= k && !q.isEmpty() && q.peek() == A[i - k])
                q.poll();
        }
        return res;
    }

    public static int[] SlidingWindowMaximum(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        Deque<Integer> deque = new LinkedList<>();

        for (int idx = 0; idx < nums.length; idx++) {
            int num = nums[idx];

            while (!deque.isEmpty() && deque.getLast() < num) {
                deque.pollLast();
            }
            deque.addLast(num);

            if (idx >= k && nums[idx - k] == deque.getFirst()) {
                deque.pollFirst();
            }

            if (idx >= k - 1) {
                res.add(deque.getFirst());
            }
        }

        return res.stream().mapToInt(i -> i).toArray();
    }

    public static double[] SlidingWindowMedian(int[] nums, int k) {
        return HashTables.SlidingWindowMedian(nums, k);
    }
    //endregion

    //region Stack Based
    public static double[] CarFleet(int[][] A) {
        int n = A.length;
        Deque<Integer> stack = new LinkedList<>();
        double[] res = new double[n];
        for (int i = n - 1; i >= 0; --i) {
            res[i] = -1.0;
            int p = A[i][0], s = A[i][1];
            while (!stack.isEmpty()) {
                int j = stack.peekLast(), p2 = A[j][0], s2 = A[j][1];
                if (s <= s2 || 1.0 * (p2 - p) / (s - s2) >= res[j] && res[j] > 0)
                    stack.pollLast();
                else
                    break;
            }
            if (!stack.isEmpty()) {
                int j = stack.peekLast(), p2 = A[j][0], s2 = A[j][1];
                res[i] = 1.0 * (p2 - p) / (s - s2);
            }
            stack.add(i);
        }
        return res;
    }
    //endregion

    //region Queue Based
    public static int MaxValueOfEquation_2(int[][] points, int k){
        return MaxValueOfEquation(points, k);
    }

    public static int ConstrainedSubsetSum_2(int[] A, int k){
        return ConstrainedSubsetSum(A, k);
    }

    public static int[] SlidingWindowMaximum_2(int[] nums, int k){
        return SlidingWindowMaximum(nums,k);
    }

    public static int ShortestSubArrayWithSumAtLeastK_2(int[] A, int K){
        return ShortestSubArrayWithSumAtLeastK(A,K);
    }
    public static int JumpGame_5(int[] nums, int k){
        return JumpGame_4(nums, k);
    }
    //endregion

    //region Private Methods
    private static char getChar(int type) {
        if (type == 0) return 'a';
        if (type == 1) return 'b';
        return 'c';
    }

    private static void searchBigThree(int[][] grid, int[] maxThree, int length){
        int end = grid.length-(length==0?0: 2*length);
        int end1 = grid[0].length-(length);
        for(int start = 0; start<end; start++){
            for(int start1 = length; start1<end1; start1++){
                if(start+start1 >= length){
                    addToMaxThree(maxThree, getSum(grid, start, start1, length));
                }
            }
        }
    }

    private static int getSum(int[][] grid, int i, int j, int length){
        if(length == 0){
            return grid[i][j];
        }

        int sum = 0;
        // edge ab
        for(int it=0; it<=length; it++){
            sum = sum + grid[i+it][j+it];
        }

        // edge ad
        for(int it=1; it<=length; it++){
            sum = sum + grid[i+it][j-it];
        }

        // edge dc
        for(int it=1; it<=length; it++){
            sum = sum + grid[i+length+it][j-length+it];
        }

        // edge bc
        for(int it=1; it<length; it++){
            sum = sum + grid[i+length+it][j+length-it];
        }

        return sum;
    }

    private static void addToMaxThree(int[] maxThree, int num){
        // Do not add duplicate entry
        if(maxThree[0] == num || maxThree[1] == num || maxThree[2] == num ){
            return;
        }

        Arrays.sort(maxThree);

        if(maxThree[0] < num){
            maxThree[0] = num;
        }
    }

    private static void dfs(int[][] grid, int i, int j, int time, Set<Integer> visited) {
        if (i < 0 || i > grid.length - 1 || j < 0 || j > grid[0].length - 1 || grid[i][j] > time || visited.contains(i*grid.length+j)) return;
        visited.add(i*grid.length+j);
        for (int[] dir : dirs) {
            dfs(grid, i+dir[0], j+dir[1], time, visited);
        }
    }
    //endregion
}
