package org.mindmaps;

import java.util.*;

import static org.helpers.Sorting.BubbleSort;

public class Array {
    //region Hashing
    public static int NumberOfGoodPairs(int[] nums) {
        int n = nums.length;
        int countOfGoodPair = 0;
        for(int i = 0; i < n; i++) {
            for(int j = i + 1; j < n; j++) {
                if(nums[i] == nums[j] && i < j) {
                    countOfGoodPair++;
                }
            }
        }

        return countOfGoodPair;
    }

    public static int CountNumberOfPairsWithAbsoluteDifferenceK(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i < j && Math.abs(nums[i] - nums[j]) == k) {
                    count++;
                }
            }
        }

        return count;
    }

    public static int CountTheNumberOfConsistentStrings(String allowed, String[] words) {
        if(allowed.isEmpty() || words.length == 0) return 0;
        List<Character> allowedCharsList = new ArrayList<>();
        int count = words.length;

        for(char a : allowed.toCharArray()) {
            allowedCharsList.add(a);
        }

        for(String word : words) {
            for(char a : word.toCharArray()) {
                if(!allowedCharsList.contains(a)){
                    count--;
                    break;
                }
            }
        }
        return count;
    }

    public static int UniqueMorseCodeWords(String[] words) {
        String[] morseCodes = new String[] {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};

        //Prepare dictionary
        Map<Character, String> morseDictionary = new HashMap<>();
        char a = 'a';
        for(String code : morseCodes) {
            morseDictionary.put(a,code);
            a++;
        }

        //Prepare MorseCode for the given words
        Map<String, String> morseCodeOfWords = new HashMap<>();
        for(String word : words) {
            StringBuilder morseCodeWord = new StringBuilder();
            for (char c : word.toCharArray()) {
                morseCodeWord.append(morseDictionary.get(c));
            }
            morseCodeOfWords.put(word, morseCodeWord.toString());
        }

        //Find the equal codes
        Map<String, Integer> countOfMorseCode = new HashMap<>();
        for(Map.Entry<String, String> morseCodeOfWord : morseCodeOfWords.entrySet()) {
            String code = morseCodeOfWord.getValue();
            countOfMorseCode.put(code, countOfMorseCode.getOrDefault(code, 0) + 1);
        }

        //Get the max count
        return countOfMorseCode.size();
    }
    //endregion

    //region Bit Manipulation
    public static int[] DecodeXORedArray(int[] encoded, int first) {
        int n = encoded.length + 1;
        int[] arr = new int[n];
        arr[0] = first;

        for (int i = 0; i < encoded.length; i++) {
            arr[i + 1] = arr[i] ^ encoded[i];
        }

        return arr;
    }

    public static int SumOfAllSubsetXORTotals(int[] nums) {
        return subsetXORSumHelper(nums, 0, 0);
    }

    public static boolean DivideArrayIntoEqualPairs(int[] nums) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();

        // Count frequency of each element
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        // Check if each frequency is even
        for (int count : frequencyMap.values()) {
            if (count % 2 != 0) {
                return false;
            }
        }

        return true;

    }

    public static int[] SetMismatch(int[] nums) {
        int n = nums.length;
        int[] result = new int[2];
        Map<Integer, Integer> countMap = new HashMap<>();

        // Populate the frequency map
        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }

        // Identify the duplicated and missing numbers
        for (int i = 1; i <= n; i++) {
            int count = countMap.getOrDefault(i, 0);
            if (count == 2) {
                result[0] = i; // duplicated number
            } else if (count == 0) {
                result[1] = i; // missing number
            }
        }

        return result;
    }
    //endregion

    //region Two Pointer
    public static int[][] FlippingAnImage_1(int[][] image) {
        int n = image.length;

        for (int i = 0; i < n; i++) {
            // Flip and invert each row simultaneously
            for (int j = 0; j < (n + 1) / 2; j++) {
                // Swap the elements and invert them
                int temp = image[i][j] ^ 1;
                image[i][j] = image[i][n - 1 - j] ^ 1;
                image[i][n - 1 - j] = temp;
            }
        }

        return image;
    }

    public static String FindFirstPalindromicStringInTheArray(String[] words) {
        for (String word : words) {
            if (isPalindrome(word)) {
                return word;
            }
        }
        return "";
    }

    public static int[] DIStringMatch(String s) {
        int n = s.length(), left = 0, right = n;
        int[] res = new int[n + 1];
        for (int i = 0; i < n; ++i)
            res[i] = s.charAt(i) == 'I' ? left++ : right--;
        res[n] = left;
        return res;
    }

    public static int[] SortArrayByParity(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int j = 0;
        Stack<Integer> oddNums = new Stack<>();
        for (int num : nums) {
            if (num % 2 == 0) {
                result[j] = num;
                j++;
            } else {
                oddNums.push(num);
            }
        }

        while(!oddNums.isEmpty()) {
            result[j] = oddNums.pop();
            j++;
        }
        return result;
    }

    public static int[] SquaresOfASortedArray(int[] nums) {
        int n = nums.length;
        //Get square
        int[] result = new int[n];
        for(int i = 0; i < n; i++) {
            result[i] = nums[i] * nums[i];
        }

        //Now return the sorted result array
        return BubbleSort(result);
    }
    //endregion

    //region Binary Search
    public static List<Integer> FindTargetIndicesAfterSortingArray(int[] nums, int target) {
        //Sort the array
        BubbleSort(nums);

        //Find the indices
        List<Integer> indices = new ArrayList<>();
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == target)
                indices.add(i);
        }
        return indices;
    }

    public static int CountNegativeNumbersInASortedMatrix_1(int[][] grid) {
        int count = 0;
        for (int[] row : grid) {
            for (int colVal : row) {
                if (colVal < 0) count++;
            }
        }
        return count;
    }

    public static int[] TheKWeakestRowsInAMatrix(int[][] mat, int k) {
        // Step 1: Calculate the number of soldiers in each row.
        int m = mat.length;
        int[] soldierCount = new int[m];
        for (int i = 0; i < m; i++) {
            soldierCount[i] = countSoldiers(mat[i]);
        }

        // Step 2: Store row indices and soldier counts.
        List<int[]> rowStrength = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            rowStrength.add(new int[]{soldierCount[i], i});
        }

        // Step 3: Sort the rows by the number of soldiers and then by row index.
        rowStrength.sort((a, b) -> {
            if (a[0] != b[0]) {
                return Integer.compare(a[0], b[0]);
            } else {
                return Integer.compare(a[1], b[1]);
            }
        });

        // Step 4: Extract the indices of the first k rows.
        int[] weakestRows = new int[k];
        for (int i = 0; i < k; i++) {
            weakestRows[i] = rowStrength.get(i)[1];
        }

        return weakestRows;
    }

    public static int SearchInsertPosition(int[] nums, int target) {
        int n = nums.length;
        int i = 0;
        while(i < n) {
            if(nums[i] == target) return i;
            else if(target > nums[i]) i++;
            else if(target < nums[i]) return i - 1;
        }
        return n;
    }

    public static boolean CheckIfNAndItsDoubleExist(int[] arr) {
        int n = arr.length;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i != j && arr[i] == 2 * arr[j]) {
                    return true;
                }
            }
        }
        return false;
    }
    //endregion Binary Search

    //region DP Based
    public static int BestTimeToBuyAndSellStock(int[] prices) {
        // Step 1: Initialize minPrice and maxProfit
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;

        // Step 2: Iterate through the prices array
        for (int price : prices) {
            // Update minPrice if the current price is lower
            if (price < minPrice) {
                minPrice = price;
            }
            // Calculate potential profit
            int profit = price - minPrice;
            // Update maxProfit if the current profit is greater
            if (profit > maxProfit) {
                maxProfit = profit;
            }
        }

        // Step 3: Return the maximum profit
        return maxProfit;
    }

    public static int GetMaximumInGeneratedArray(int n) {
        // Step 1: Handle edge cases
        if (n == 0) return 0;
        if (n == 1) return 1;

        // Step 2: Initialize the array
        int[] nums = new int[n + 1];
        nums[0] = 0;
        nums[1] = 1;

        int max = 1; // Since nums[1] = 1, start max as 1

        // Step 3: Populate the array
        for (int i = 1; 2 * i <= n; i++) {
            nums[2 * i] = nums[i];
            if (2 * i + 1 <= n) {
                nums[2 * i + 1] = nums[i] + nums[i + 1];
                max = Math.max(max, nums[2 * i + 1]);
            }
        }

        // Step 4: Return the maximum value in the array
        return max;
    }

    public static int MaximumSubArray(int[] nums) {
        // Step 1: Initialize maxCurrent and maxGlobal
        int maxCurrent = nums[0];
        int maxGlobal = nums[0];

        // Step 2: Iterate through the array starting from the second element
        for (int i = 1; i < nums.length; i++) {
            // Update maxCurrent to be the maximum of the current element or the current element plus the previous maxCurrent
            maxCurrent = Math.max(nums[i], maxCurrent + nums[i]);

            // Update maxGlobal to be the maximum of maxGlobal and maxCurrent
            if (maxCurrent > maxGlobal) {
                maxGlobal = maxCurrent;
            }
        }

        // Step 3: Return the result
        return maxGlobal;
    }

    public static int StoneGame2(int[] piles){
        int n = piles.length;

        // Prefix sum array to calculate the sum of remaining piles efficiently
        int[] suffixSum = new int[n];
        suffixSum[n - 1] = piles[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            suffixSum[i] = suffixSum[i + 1] + piles[i];
        }

        // Memoization table
        int[][] dp = new int[n][n];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }

        return stoneGameHelper(piles, dp, suffixSum, 0, 1);
    }

    public static int BeautifulArrangement(int n) {
        boolean[] visited = new boolean[n + 1];
        return countOfBeautifulArrangement(n, 1, visited);
    }
    //endregion

    //region Sliding Window
    public static int MinimumDifferenceBetweenHighestAndLowestOfKScores(int[] nums, int k) {
        Arrays.sort(nums);
        int minDifference = Integer.MAX_VALUE;
        for (int i = 0; i <= nums.length - k; i++) {
            int difference = nums[i + k - 1] - nums[i];
            minDifference = Math.min(minDifference, difference);
        }
        return minDifference;
    }

    public static double MaximumAverageSubArray_1(int[] nums, int k) {
        double maxSum = 0;
        for (int i = 0; i < k; i++) {
            maxSum += nums[i];
        }

        double currentSum = maxSum;

        for (int i = k; i < nums.length; i++) {
            currentSum += nums[i] - nums[i - k];
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum / k;
    }

    public static int MaximumPointsYouCanObtainFromCards(int[] cardPoints, int k) {
        int n = cardPoints.length;
        // Calculate the total sum of the array
        int totalSum = 0;
        for (int points : cardPoints) {
            totalSum += points;
        }

        // Edge case: If k equals n, take all cards
        if (k == n) {
            return totalSum;
        }

        // Find the minimum sum of the subarray of length n - k
        int windowSize = n - k;
        int currentWindowSum = 0;

        // Initial sum of the first window
        for (int i = 0; i < windowSize; i++) {
            currentWindowSum += cardPoints[i];
        }

        int minWindowSum = currentWindowSum;

        // Slide the window across the array
        for (int i = windowSize; i < n; i++) {
            currentWindowSum += cardPoints[i] - cardPoints[i - windowSize];
            minWindowSum = Math.min(minWindowSum, currentWindowSum);
        }

        // The maximum score is totalSum minus the minimum window sum
        return totalSum - minWindowSum;
    }

    public static int BinarySubArraysWithSum(int[] nums, int goal) {
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        prefixSumCount.put(0, 1);
        int prefixSum = 0;
        int result = 0;
        for (int num : nums) {
            prefixSum += num;
            result += prefixSumCount.getOrDefault(prefixSum - goal, 0);
            prefixSumCount.put(prefixSum, prefixSumCount.getOrDefault(prefixSum, 0) + 1);
        }

        return result;
    }

    public static int MinimumSwapsToGroupAllOnesTogether_2(int[] nums) {
        int n = nums.length;

        //Count ones
        int totalOnes = 0;
        for (int num : nums) {
            if (num == 1) {
                totalOnes++;
            }
        }

        //If all element is one then no swap
        if (totalOnes == 0 || totalOnes == n) {
            return 0;
        }

        //1. Create a sliding window of size totalOnes over the doubled array
        int[] doubled = new int[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            doubled[i] = nums[i % n];
        }

        //2. Calculate minimum swaps needed
        int currentZeros = 0;

        //2.1 Calculate number of 0's in the first window of size totalOnes
        for (int i = 0; i < totalOnes; i++) {
            if (doubled[i] == 0) {
                currentZeros++;
            }
        }

        //2.2 Assume it as minimum number of swaps
        int minSwaps = currentZeros;

        //2.3 Slide the window across the doubled array
        for (int i = totalOnes; i < 2 * n; i++) {
            if (doubled[i] == 0) {
                currentZeros++;
            }
            if (doubled[i - totalOnes] == 0) {
                currentZeros--;
            }
            minSwaps = Math.min(minSwaps, currentZeros);
        }

        return minSwaps;
    }
    //endregion

    //region Geometry
    public static boolean CheckIfItIsAStraightLine(int[][] coordinates) {
        // There must be at least two points to form a line
        if (coordinates.length < 2) {
            return false;
        }

        // Calculate the slope difference using cross product
        int x1 = coordinates[0][0], y1 = coordinates[0][1];
        int x2 = coordinates[1][0], y2 = coordinates[1][1];
        int dx = x2 - x1;
        int dy = y2 - y1;

        for (int i = 2; i < coordinates.length; i++) {
            int x3 = coordinates[i][0], y3 = coordinates[i][1];
            int dx2 = x3 - x2;
            int dy2 = y3 - y2;

            // Check if cross products are equal
            if (dy * dx2 != dy2 * dx) {
                return false;
            }
        }

        return true;
    }

    public static int[] QueriesOnNumberOfPointsInsideACircle(int[][] points, int[][] queries) {
        int[] answer = new int[queries.length];

        // Iterate through each query
        for (int j = 0; j < queries.length; j++) {
            int xj = queries[j][0];
            int yj = queries[j][1];
            int rj = queries[j][2];
            int rjSquared = rj * rj;

            int count = 0;

            // Check each point if it lies within the current circle
            for (int[] point : points) {
                int xi = point[0];
                int yi = point[1];

                // Calculate the squared distance from the point to the center of the circle
                int dx = xi - xj;
                int dy = yi - yj;
                int distanceSquared = dx * dx + dy * dy;

                // If the distance is within the radius, increment the count
                if (distanceSquared <= rjSquared) {
                    count++;
                }
            }

            // Store the result for the current query
            answer[j] = count;
        }

        return answer;
    }

    public static int DetonateTheMaximumBombs(int[][] bombs) {
        int n = bombs.length;
        List<List<Integer>> graph = new ArrayList<>();

        // Step 1: Construct the graph
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    long dx = bombs[i][0] - bombs[j][0];
                    long dy = bombs[i][1] - bombs[j][1];
                    long distanceSquared = dx * dx + dy * dy;
                    long radiusSquared = (long) bombs[i][2] * bombs[i][2];

                    if (distanceSquared <= radiusSquared) {
                        graph.get(i).add(j);
                    }
                }
            }
        }

        // Step 2: Find the maximum number of bombs that can be detonated
        int maxDetonations = 0;
        for (int i = 0; i < n; i++) {
            boolean[] visited = new boolean[n];
            maxDetonations = Math.max(maxDetonations, detonateBombHelper(graph, visited, i));
        }

        return maxDetonations;
    }

    public static int MaxPointsOnALine(int[][] points) {
        if (points.length < 2) {
            return points.length;
        }

        int maxPoints = 1;

        for (int i = 0; i < points.length; i++) {
            Map<String, Integer> slopeMap = new HashMap<>();
            int duplicate = 0;
            int currentMax = 0;

            for (int j = i + 1; j < points.length; j++) {
                int dx = points[j][0] - points[i][0];
                int dy = points[j][1] - points[i][1];

                if (dx == 0 && dy == 0) {
                    duplicate++;
                    continue;
                }

                int gcd = maxPointsOnALineHelper(dx, dy);
                dx /= gcd;
                dy /= gcd;

                String slopeKey = dx + "/" + dy;
                slopeMap.put(slopeKey, slopeMap.getOrDefault(slopeKey, 0) + 1);
                currentMax = Math.max(currentMax, slopeMap.get(slopeKey));
            }

            maxPoints = Math.max(maxPoints, currentMax + duplicate + 1);
        }

        return maxPoints;
    }
    //endregion

    //region Matrix
    public static int RichestCustomerWealth(int[][] accounts) {
        int maxWealth = 0;

        // Iterate over each customer
        for (int[] customer : accounts) {
            int wealth = 0;

            // Calculate the wealth of the current customer
            for (int account : customer) {
                wealth += account;
            }

            // Update the maximum wealth if the current customer's wealth is higher
            if (wealth > maxWealth) {
                maxWealth = wealth;
            }
        }

        return maxWealth;
    }

    public static int[][] FlippingAnImage_2(int[][] image) {
        return FlippingAnImage_1(image);
    }

    public static int MatrixDiagonalSum(int[][] mat) {
        int n = mat.length;
        int primaryDiagonalSum = 0;
        int secondaryDiagonalSum = 0;

        for (int i = 0; i < n; i++) {
            primaryDiagonalSum += mat[i][i];
            secondaryDiagonalSum += mat[i][n - 1 - i];
        }

        // If n is odd, subtract the middle element as it has been counted twice
        if (n % 2 == 1) {
            primaryDiagonalSum -= mat[n / 2][n / 2];
        }

        return primaryDiagonalSum + secondaryDiagonalSum;
    }

    public static int CountNegativeNumbersInASortedMatrix_2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int count = 0;

        int row = 0;
        int col = n - 1;

        while (row < m && col >= 0) {
            if (grid[row][col] < 0) {
                // All elements in this column from row to the end are negative
                count += (m - row);
                // Move left
                col--;
            } else {
                // Move down
                row++;
            }
        }

        return count;
    }

    //region Sub-Rectangle Queries
    private static int[][] matrix;
    private static int[][] updates;
    private static int updateCount;

    public static void SubRectangleQueries(int[][] rectangle) {
        matrix = rectangle;
        updates = new int[rectangle.length][rectangle[0].length];
        updateCount = 0;
    }

    public static void UpdateSubRectangle(int row1, int col1, int row2, int col2, int newValue) {
        updateCount++;
        for (int i = row1; i <= row2; i++) {
            for (int j = col1; j <= col2; j++) {
                updates[i][j] = updateCount;
                matrix[i][j] = newValue;
            }
        }
    }

    public static int GetSubRectangleValue(int row, int col) {
        // Find the most recent update affecting this cell
        if (updates[row][col] == updateCount) {
            return matrix[row][col];
        } else {
            return matrix[row][col];
        }
    }
    //endregion

    //endregion

    //region Privates
    private static int subsetXORSumHelper(int[] nums, int index, int currentXOR) {
        if (index == nums.length) {
            return currentXOR;
        }

        // Include nums[index] in the subset
        int withCurrent = subsetXORSumHelper(nums, index + 1, currentXOR ^ nums[index]);

        // Exclude nums[index] from the subset
        int withoutCurrent = subsetXORSumHelper(nums, index + 1, currentXOR);

        return withCurrent + withoutCurrent;
    }

    private static boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    private static int countSoldiers(int[] row) {
        int count = 0;
        for (int soldier : row) {
            if (soldier == 1) {
                count++;
            } else {
                break;
            }
        }
        return count;
    }

    private static int stoneGameHelper(int[] piles, int[][] dp, int[] suffixSum, int i, int M) {
        int n = piles.length;

        // Base case: If we've reached or passed the end of the piles
        if (i >= n) return 0;

        // If all piles can be taken, return the sum of the remaining piles
        if (2 * M >= n - i) return suffixSum[i];

        // Check if the result is already calculated
        if (dp[i][M] != -1) return dp[i][M];

        int maxStones = 0;

        // Try taking X piles where 1 <= X <= 2 * M
        for (int X = 1; X <= 2 * M; X++) {
            int nextM = Math.max(M, X);
            int stonesTaken = suffixSum[i] - stoneGameHelper(piles, dp, suffixSum, i + X, nextM);
            maxStones = Math.max(maxStones, stonesTaken);
        }

        // Store the result in the memoization table
        dp[i][M] = maxStones;
        return maxStones;
    }

    private static int countOfBeautifulArrangement(int n, int pos, boolean[] visited) {
        if (pos > n) {
            return 1;  // A valid arrangement is found
        }

        int count = 0;
        for (int i = 1; i <= n; i++) {
            // Check if the number i can be placed in position pos
            if (!visited[i] && (i % pos == 0 || pos % i == 0)) {
                visited[i] = true;  // Mark the number as used
                count += countOfBeautifulArrangement(n, pos + 1, visited);  // Recurse for the next position
                visited[i] = false;  // Backtrack
            }
        }

        return count;
    }

    private static int detonateBombHelper(List<List<Integer>> graph, boolean[] visited, int i) {
        visited[i] = true;
        int count = 1;

        for (int neighbor : graph.get(i)) {
            if (!visited[neighbor]) {
                count += detonateBombHelper(graph, visited, neighbor);
            }
        }

        return count;
    }

    private static int maxPointsOnALineHelper(int a, int b) {
        if (b == 0) {
            return a;
        }
        return maxPointsOnALineHelper(b, a % b);
    }
    //endregion
}
