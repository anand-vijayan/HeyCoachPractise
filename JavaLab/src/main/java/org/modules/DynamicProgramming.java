package org.modules;

import java.math.BigInteger;
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

    public static int MaxCoins_2(int[] numbers) {
        if (numbers.length == 1) return numbers[0];  // Only one house, return its value
        if (numbers.length == 2) return Math.max(numbers[0], numbers[1]);  // Two houses, return the max

        // Calculate the max rob value by excluding the last house and the first house respectively
        return Math.max(MaxCoin2Helper(numbers, 0, numbers.length - 2), MaxCoin2Helper(numbers, 1, numbers.length - 1));
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

    public static int MinimumJumps_1(int[] v, int n){
        if (n == 0 || v[0] == 0) return -1;
        if (n == 1) return 0;

        // Initialize the BFS queue
        Queue<Integer> queue = new LinkedList<>();
        queue.add(0);

        // Initialize the distance array
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
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
    public static int GridPuzzle_1(int m, int n){
        return BinomialCoefficient(m + n - 2, m - 1);
    }

    public static int GridPuzzle_2(int n, int m, int[][] obstacleGrid) {

        // If the start or the end is an obstacle, return 0.
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1) {
            return 0;
        }

        // DP table to store number of ways to reach each cell.
        int[][] dp = new int[m][n];

        // Initialize the starting point.
        dp[0][0] = 1;

        // Fill the first column.
        for (int i = 1; i < m; i++) {
            dp[i][0] = (obstacleGrid[i][0] == 1) ? 0 : dp[i-1][0];
        }

        // Fill the first row.
        for (int j = 1; j < n; j++) {
            dp[0][j] = (obstacleGrid[0][j] == 1) ? 0 : dp[0][j-1];
        }

        // Fill the rest of the DP table.
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;  // Blocked cell.
                } else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];  // Sum of top and left cells.
                }
            }
        }

        // The bottom-right corner contains the result.
        return dp[m-1][n-1];
    }

    public static int MinTaxPaid_1(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        // DP array to store the minimum cost to reach each cell.
        int[][] dp = new int[m][n];

        // Initialize the starting point.
        dp[0][0] = grid[0][0];

        // Fill the first row.
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }

        // Fill the first column.
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }

        // Fill the rest with the DP table.
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }

        // The bottom-right corner contains the result.
        return dp[m-1][n-1];
    }

    public static int TheNewClassmate(int n, String s) {
        // Initialize a DP array where dp[i] stores the length of the longest
        // lexicographically increasing subsequence ending at index i.
        int[] dp = new int[n];

        // Every character on its own forms a subsequence of length 1.
        Arrays.fill(dp, 1);

        // Call the helper method to calculate the longest increasing subsequence
        return TheNewClassmateHelper(n, s, dp);
    }

    public static int CrossingTheJungle(int n, int m, int[][] grid) {
        //To:DO

        // Return the minimum initial experience required
        return grid[0][0];
    }
    //endregion

    //region Dynamic Programming 3
    public static int MinTaxPaid_2(int[][] grid, int m, int n) {
        n = grid.length;
        int[][] dp = new int[n][n];

        // Initialize dp with the grid's first cell
        dp[0][0] = grid[0][0];

        // Fill first row (can only come from the left)
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }

        // Fill first column (can only come from above)
        for (int i = 1; i < n; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }

        // Fill the rest of the dp table
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j],
                        Math.min(dp[i][j - 1], dp[i - 1][j - 1]));
            }
        }

        // The answer is in the bottom-right corner of the dp table
        return dp[n - 1][n - 1];
    }

    public static int PartitionWithMinimumSumDifference(int[] arr, int n) {
        n = arr.length;
        int totalSum = 0;

        // Calculate total sum of all elements
        for (int num : arr) {
            totalSum += num;
        }

        int target = totalSum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;  // Base case: 0 sum is always possible

        // Update the dp array
        for (int num : arr) {
            for (int j = target; j >= num; j--) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        // Find the maximum sum <= totalSum / 2 that can be formed
        int closestSum = 0;
        for (int j = target; j >= 0; j--) {
            if (dp[j]) {
                closestSum = j;
                break;
            }
        }

        // Return the minimum absolute difference
        return totalSum - 2 * closestSum;
    }

    public static boolean SumEqualToK(int[] arr, int n, int k) {
        // Create a DP array to store results
        boolean[][] dp = new boolean[n + 1][k + 1];

        // Base case: sum of 0 is possible with an empty subset
        for (int i = 0; i <= n; i++) {
            dp[i][0] = true;
        }

        // Fill the DP table
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k; j++) {
                // Exclude the current element
                dp[i][j] = dp[i-1][j];

                // Include the current element if it's not greater than the current sum
                if (arr[i - 1] <= j) {
                    dp[i][j] = dp[i][j] || dp[i-1][j - arr[i - 1]];
                }
            }
        }

        // Return the result stored in dp[N][K]
        return dp[n][k];
    }

    public static int MaximumFallingPathSum_2(int[][] grid) {
        int n = grid.length;
        int[][] dp = new int[n][n];

        // Initialize the dp table with the first row of the grid
        System.arraycopy(grid[0], 0, dp[0], 0, n);

        // Iterate through each row starting from the second
        for (int i = 1; i < n; i++) {
            // Precompute the maximum and second maximum from the previous row
            int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE;
            for (int j = 0; j < n; j++) {
                if (dp[i-1][j] > max1) {
                    max2 = max1;
                    max1 = dp[i-1][j];
                } else if (dp[i-1][j] > max2) {
                    max2 = dp[i-1][j];
                }
            }

            // Fill the dp table for the current row
            for (int j = 0; j < n; j++) {
                // If the current column had the maximum in the previous row, use the second maximum
                if (dp[i-1][j] == max1) {
                    dp[i][j] = grid[i][j] + max2;
                } else {
                    dp[i][j] = grid[i][j] + max1;
                }
            }
        }

        // The answer is the maximum value in the last row
        int result = Integer.MIN_VALUE;
        for (int j = 0; j < n; j++) {
            result = Math.max(result, dp[n-1][j]);
        }

        return result;
    }

    public static int EggDropping(int n, int k) {
        // Create a DP table where dp[i][j] will represent minimum number of moves
        // needed with i eggs and j floors
        int[][] dp = new int[n + 1][k + 1];

        // Base case: when we have one egg, we need to try every floor from 1 to K
        for (int j = 1; j <= k; j++) {
            dp[1][j] = j;
        }

        // Base case: if there are 0 floors, no trials are needed. If there is 1 floor, one trial is needed.
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 0;
            dp[i][1] = 1;
        }

        // Fill the rest of the table using the recursive relation
        for (int i = 2; i <= n; i++) {
            for (int j = 2; j <= k; j++) {
                dp[i][j] = Integer.MAX_VALUE;

                // Perform binary search to optimize the minimum number of moves
                int low = 1, high = j;
                while (low <= high) {
                    int mid = (low + high) / 2;
                    int eggBreaks = dp[i-1][mid-1];  // Egg breaks
                    int eggDoesntBreak = dp[i][j-mid];  // Egg doesn't break

                    int worstCase = 1 + Math.max(eggBreaks, eggDoesntBreak);

                    dp[i][j] = Math.min(dp[i][j], worstCase);

                    // Use binary search to optimize the drop
                    if (eggBreaks > eggDoesntBreak) {
                        high = mid - 1;
                    } else {
                        low = mid + 1;
                    }
                }
            }
        }

        // Return the result for N eggs and K floors
        return dp[n][k];
    }
    //endregion

    //region Dynamic Programming 4
    public static int StolenGifts(int[] gifts, int n, int k) {
        // Use a set to store the unique combinations of values
        Set<List<Integer>> uniqueCombinations = new HashSet<>();

        // Sort the gift array to help with skipping duplicates
        Arrays.sort(gifts);

        // Start backtracking from index 0
        StolenGiftsHelper(gifts, k, 0, new ArrayList<>(), uniqueCombinations);

        // Return the number of unique combinations
        return uniqueCombinations.size();
    }

    public static int PartitionWithGivenDifference(int n, int d, int[] nums) {
        int total_sum = 0;
        for (int num : nums) {
            total_sum += num;
        }

        // Check if the partition is possible
        if ((total_sum + d) % 2 != 0 || (total_sum - d) % 2 != 0) {
            return 0;
        }

        int sum1 = (total_sum + d) / 2;
        if (sum1 < 0 || sum1 > total_sum) {
            return 0;
        }

        // Initialize the DP array
        int[] dp = new int[sum1 + 1];
        dp[0] = 1;

        // Fill the DP array
        for (int num : nums) {
            for (int j = sum1; j >= num; j--) {
                dp[j] = (dp[j] + dp[j - num]) % MOD;
            }
        }

        return dp[sum1];
    }

    public static int Knapsack(int[] wt, int[] val, int n, int W) {
        // Initialize a DP array with 0s
        int[] dp = new int[W + 1];

        // Fill the DP array
        for (int i = 0; i < n; i++) {
            // Process weights from W down to wt[i]
            for (int j = W; j >= wt[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - wt[i]] + val[i]);
            }
        }

        // The result is the maximum value for knapsack capacity W
        return dp[W];
    }

    public static int LongestPalindromicExams(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];

        // Base case: single character substrings are palindromes of length 1
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }

        // Build the DP table
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[0][n - 1];
    }

    public static int BreakTheNumber(int n){
        if (n == 2) return 1; // Special case for n = 2
        if (n == 3) return 2; // Special case for n = 3

        // For n >= 4
        int product = 1;
        while (n > 4) {
            product *= 3;
            n -= 3;
        }
        product *= n; // Multiply the remaining part
        return product;
    }
    //endregion

    //region Dynamic Programming 5
    public static int MinimumCoinsRequired_1(int[] coins, int amount){
        // Edge case
        if (amount == 0) return 0;

        // Initialize dp array with a large value (amount + 1 is an upper bound)
        int[] dp = new int[amount + 1];
        for (int i = 1; i <= amount; i++) {
            dp[i] = amount + 1; // Set initial value as a large number
        }

        // Base case: 0 amount requires 0 coins
        dp[0] = 0;

        // Process each amount from 1 to target amount
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i >= coin) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }

        // If dp[amount] is still a large number, return -1, else return the result
        return dp[amount] > amount ? -1 : dp[amount];
    }

    public static int MinimumCoinsRequired_2(int[] coins,int n, int amount){
        // Edge case: when amount is 0
        if (amount == 0) return 1;

        // Initialize dp array with 0
        int[] dp = new int[amount + 1];
        dp[0] = 1; // There's one way to make 0 amount

        // Process each coin
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }

        // Return the number of combinations to make the given amount
        return dp[amount];
    }

    public static String longestCommonSubstring(String str1, String str2) {
        int m = str1.length();
        int n = str2.length();

        // Initialize DP table
        int[][] dp = new int[m + 1][n + 1];
        int maxLength = 0;
        int endIndexInStr1 = 0;

        // Fill DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > maxLength) {
                        maxLength = dp[i][j];
                        endIndexInStr1 = i - 1;
                    }
                }
            }
        }

        // Extract the longest common substring
        if (maxLength == 0) return "";
        return str1.substring(endIndexInStr1 - maxLength + 1, endIndexInStr1 + 1);
    }

    public static int MinimumJumps_2(int[] arr, int n) {
        n = arr.length;

        if (n == 0 || (n > 1 && arr[0] == 0)) {
            // Cannot move if array is empty or starting point is 0
            return -1;
        }

        int jumps = 0;
        int currentEnd = 0;
        int farthest = 0;

        for (int i = 0; i < n - 1; i++) {
            farthest = Math.max(farthest, i + arr[i]);

            if (i == currentEnd) {
                jumps++;
                currentEnd = farthest;

                if (currentEnd >= n - 1) {
                    // Reach or exceed the last index
                    return jumps;
                }
            }
        }

        // If we exit the loop without reaching the end
        return -1;
    }

    public static int LongestConsecutiveSequence(int[] numbers) {
        if (numbers.length == 0) return 0;

        Set<Integer> numSet = new HashSet<>();
        for (int num : numbers) {
            numSet.add(num);
        }

        int longestStreak = 0;

        for (int num : numSet) {
            // Check if it's the start of a sequence
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                // Count the length of the consecutive sequence
                while (numSet.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                // Update the longest streak found
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }
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
        if (S.isEmpty()) {
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
        if (s == null || s.isEmpty()) return 0;

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

    private static int BinomialCoefficient(int n, int k) {
        BigInteger res = BigInteger.ONE;
        for (int i = 0; i < k; i++) {
            res = res.multiply(BigInteger.valueOf(n - i)).divide(BigInteger.valueOf(i + 1));
        }
        return res.intValue();
    }

    private static int TheNewClassmateHelper(int n, String s, int[] dp) {
        // Loop through each character of the string starting from the second character.
        for (int i = 1; i < n; i++) {
            // For each character, compare it with previous characters.
            for (int j = i - 1; j >= 0; j--) {
                // Only consider increasing sequences, so skip if s[j] > s[i].
                if (s.charAt(j) > s.charAt(i)) {
                    continue;
                }

                // Calculate the possible new subsequence length by extending the sequence ending at j.
                int ans = dp[j] + 1;

                // If this new sequence length is greater than the current value at dp[i], update dp[i].
                if (ans > dp[i]) {
                    dp[i] = ans;
                }
            }
        }

        // Find the best (maximum) value in the dp array, which gives the length of the longest
        // lexicographically increasing subsequence.
        int best = 1;
        for (int i = 0; i < n; i++) {
            if (best < dp[i]) {
                best = dp[i];
            }
        }

        return best;
    }

    private static void StolenGiftsHelper(int[] gifts, int target, int start, List<Integer> current, Set<List<Integer>> uniqueCombinations) {
        // Base case: if target is reached, add the current combination to the set
        if (target == 0) {
            uniqueCombinations.add(new ArrayList<>(current));
            return;
        }

        // Explore all possible combinations starting from the current index
        for (int i = start; i < gifts.length; i++) {
            if (gifts[i] > target) {
                // If the current gift exceeds the target, break out (since the array is sorted)
                break;
            }

            // Skip duplicates by checking if this is the same as the previous element
            if (i > start && gifts[i] == gifts[i - 1]) {
                continue;
            }

            // Include the current gift in the combination and continue exploring
            current.add(gifts[i]);
            StolenGiftsHelper(gifts, target - gifts[i], i + 1, current, uniqueCombinations);
            current.remove(current.size() - 1);  // Backtrack
        }
    }

    //endregion

    //region Variables & Constants
    private static final int MOD = 1000000007;
    //endregion
}
