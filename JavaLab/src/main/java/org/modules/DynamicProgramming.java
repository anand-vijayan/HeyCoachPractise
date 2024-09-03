package org.modules;

import java.util.*;

public class DynamicProgramming {

    //region Backtracking and Recursion
    public static int IngredientsOfAlchemist(int n, int ingredientsPerDay) {
        if(n == 1) {
            return ingredientsPerDay;
        }
        return ingredientsPerDay + IngredientsOfAlchemist(n-1, ingredientsPerDay);
    }

    public static int BalancedParenthesis(int n) {
        List<String> result = new ArrayList<>();
        GenerateParenthesesHelper(result, "", n, n);
        return result.size();
    }

    public static void BuntyInTheHorrorHouse(int n, int[][] arr) {
        System.out.println(BuntyInTheHorrorHouseHelper(arr,0,0,n));
    }

    public static int SentenceCount(int n, List<String> dict, String S){
        Map<String, Integer> memo = new HashMap<>();
        return SentenceCountHelper(S, dict, memo);
    }

    public static int FairDistributionOfCookies(int[] cookies, int k) {
        int[] children = new int[k];
        return FairDistributionOfCookiesHelper(cookies, children, 0);
    }

    //endregion

    //region Dynamic Programming 1
    public static int MaxCoins_1(int n, int[] arr) {
        if (n == 0) return 0;
        if (n == 1) return arr[0];
        if (n == 2) return Math.max(arr[0], arr[1]);

        int[] dp = new int[n];
        dp[0] = arr[0];
        dp[1] = Math.max(arr[0], arr[1]);

        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2] + arr[i]);
        }

        return dp[n-1];
    }

    public static int MaxCoins_2(int[] nums) {
        if (nums.length == 1) return nums[0];  // Only one house, return its value
        if (nums.length == 2) return Math.max(nums[0], nums[1]);  // Two houses, return the max

        // Calculate the max rob value by excluding the last house and the first house respectively
        return Math.max(MaxCoin2Helper(nums, 0, nums.length - 2), MaxCoin2Helper(nums, 1, nums.length - 1));
    }

    public static int GettingInFaangCompanies(int n, int[][] points) {
        int[][] dp = new int[n][3];

        // Initialize the first day's points
        dp[0][0] = points[0][0];
        dp[0][1] = points[0][1];
        dp[0][2] = points[0][2];

        // Fill dp array for subsequent days
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i-1][1], dp[i-1][2]) + points[i][0];
            dp[i][1] = Math.max(dp[i-1][0], dp[i-1][2]) + points[i][1];
            dp[i][2] = Math.max(dp[i-1][0], dp[i-1][1]) + points[i][2];
        }

        // The maximum merit points achievable will be the max value on the last day
        return Math.max(dp[n-1][0], Math.max(dp[n-1][1], dp[n-1][2]));
    }

    public static int MinimumJumps(int[] v, int n){
        if (n == 0 || v[0] == 0) return -1;
        if (n == 1) return 0;

        // Initialize the BFS queue
        Queue<Integer> queue = new LinkedList<>();
        queue.add(0);

        // Initialize the distance array
        int[] dist = new int[n];
        for (int i = 0; i < n; i++) {
            dist[i] = Integer.MAX_VALUE;
        }
        dist[0] = 0;

        // BFS traversal
        while (!queue.isEmpty()) {
            int current = queue.poll();

            // Find the farthest reachable index from current position
            int maxReach = Math.min(n - 1, current + v[current]);
            for (int i = current + 1; i <= maxReach; i++) {
                if (dist[i] == Integer.MAX_VALUE) { // Not visited
                    dist[i] = dist[current] + 1;
                    queue.add(i);

                    // If we reach the end, return the number of jumps
                    if (i == n - 1) return dist[i];
                }
            }
        }

        // If we exhaust the queue and haven't reached the end
        return -1;
    }

    public static int SecretCodes(int n, int[] ar) {
        StringBuilder sb = new StringBuilder();
        for (int num : ar) {
            sb.append(num);
        }
        String s = sb.toString();

        return SecretCodeHelper(s);
    }
    //endregion

    //region Dynamic Programming 2
    //endregion

    //region Private
    private static void GenerateParenthesesHelper(List<String> result, String current, int open, int close) {
        // If no more open or close parentheses to add, add the current combination to result
        if (open == 0 && close == 0) {
            result.add(current);
            return;
        }

        // If there are open parentheses remaining, add an open parenthesis
        if (open > 0) {
            GenerateParenthesesHelper(result, current + "(", open - 1, close);
        }

        // If there are more closing than opening parentheses remaining, add a close parenthesis
        if (close > open) {
            GenerateParenthesesHelper(result, current + ")", open, close - 1);
        }
    }

    private static int BuntyInTheHorrorHouseHelper(int[][] maze, int x, int y, int n) {
        // Base case: If Bunty reaches the exit
        if (x == n - 1 && y == n - 1 && maze[x][y] == 1) {
            return 1;
        }

        // Check if the current position is within the maze and accessible
        if (x < 0 || y < 0 || x >= n || y >= n || maze[x][y] == 0) {
            return 0;
        }

        // Mark the current cell as visited by setting it to 0 to avoid revisiting
        maze[x][y] = 0;

        // Explore all four possible directions: right, left, down, up
        int totalPaths = BuntyInTheHorrorHouseHelper(maze, x + 1, y, n)  // Move down
                + BuntyInTheHorrorHouseHelper(maze, x - 1, y, n)  // Move up
                + BuntyInTheHorrorHouseHelper(maze, x, y + 1, n)  // Move right
                + BuntyInTheHorrorHouseHelper(maze, x, y - 1, n); // Move left

        // Backtrack: Unmark the current cell as visited
        maze[x][y] = 1;

        return totalPaths;
    }

    private static int SentenceCountHelper(String S, List<String> dict, Map<String, Integer> memo){
        // Base case: If the string is empty, one valid sentence is found
        if (S.length() == 0) {
            return 1;
        }

        // If the result for this substring is already computed, return it
        if (memo.containsKey(S)) {
            return memo.get(S);
        }

        int count = 0;

        // Try every possible prefix of the string
        for (int i = 1; i <= S.length(); i++) {
            String prefix = S.substring(0, i);
            if (dict.contains(prefix)) {
                // If prefix is a valid word, recursively count sentences for the remaining string
                String suffix = S.substring(i);
                count += SentenceCountHelper(suffix, dict, memo);
            }
        }

        // Store the result in the memoization map
        memo.put(S, count);

        return count;
    }

    private static int FairDistributionOfCookiesHelper(int[] cookies, int[] children, int index) {
        // Base case: If we've assigned all cookies, return the unfairness (max cookies received by any child)
        if (index == cookies.length) {
            int maxCookies = 0;
            for (int child : children) {
                maxCookies = Math.max(maxCookies, child);
            }
            return maxCookies;
        }

        int minUnfairness = Integer.MAX_VALUE;

        // Try giving the current bag of cookies to each child and find the minimum unfairness
        for (int i = 0; i < children.length; i++) {
            // Assign cookies[index] to child i
            children[i] += cookies[index];
            // Recur to assign the next bag of cookies
            minUnfairness = Math.min(minUnfairness, FairDistributionOfCookiesHelper(cookies, children, index + 1));
            // Backtrack: undo the assignment
            children[i] -= cookies[index];
        }

        return minUnfairness;
    }

    private static int MaxCoin2Helper(int[] nums, int start, int end) {
        int prev2 = 0, prev1 = 0, curr = 0;

        for (int i = start; i <= end; i++) {
            curr = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }

        return curr;
    }

    private static int SecretCodeHelper(String s) {
        if (s == null || s.length() == 0) return 0;

        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;

        // Base case for the first character
        dp[1] = (s.charAt(0) != '0') ? 1 : 0;

        for (int i = 2; i <= n; i++) {
            // Check if the last single character is valid
            int oneDigit = Integer.parseInt(s.substring(i - 1, i));
            if (oneDigit >= 1 && oneDigit <= 9) {
                dp[i] += dp[i - 1];
            }

            // Check if the last two characters form a valid letter
            int twoDigits = Integer.parseInt(s.substring(i - 2, i));
            if (twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i - 2];
            }
        }

        return dp[n];
    }
    //endregion
}
