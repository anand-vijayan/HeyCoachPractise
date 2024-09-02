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
            String morseCodeWord = "";
            for (char c : word.toCharArray()) {
                morseCodeWord += morseDictionary.get(c);
            }
            morseCodeOfWords.put(word,morseCodeWord);
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
    public static int[][] FlippingAnImage(int[][] image) {
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
        for(int i = 0; i < n; i++) {
            if(nums[i]%2 == 0) {
                result[j] = nums[i];
                j++;
            } else {
                oddNums.push(nums[i]);
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
        nums = BubbleSort(nums);

        //Find the indices
        List<Integer> indices = new ArrayList<>();
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == target)
                indices.add(i);
        }
        return indices;
    }

    public static int CountNegativeNumbersInASortedMatrix(int[][] grid) {
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
        Collections.sort(rowStrength, (a, b) -> {
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
    //endregion
}
