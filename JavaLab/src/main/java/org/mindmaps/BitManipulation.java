package org.mindmaps;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BitManipulation {

    //region Variables & Constants
    private static int bobPoint = 0;
    private static int[] maxbob = new int[12];
    //endregion

    //region Simple Bit Manipulation
    public static int NumberOfStepsToReduceANumberToZero(int num) {
        int steps = 0;
        while (num > 0) {
            if (num % 2 == 0) {
                num /= 2; // Divide by 2 if the number is even
            } else {
                num -= 1; // Subtract 1 if the number is odd
            }
            steps++;
        }
        return steps;
    }

    public static int[] DecodeXORedArray(int[] encoded, int first) {
        int n = encoded.length + 1;
        int[] arr = new int[n];
        arr[0] = first; // Set the first element of the array

        for (int i = 0; i < encoded.length; i++) {
            arr[i + 1] = arr[i] ^ encoded[i]; // Decode using XOR operation
        }

        return arr;
    }

    public static int XOROperationInAnArray(int n, int start) {
        int result = 0;
        for (int i = 0; i < n; i++) {
            result ^= start + 2 * i; // XOR each element in the array
        }
        return result;
    }

    public static int CountTheNumberOfConsistentStrings_1(String allowed, String[] words) {
        return Strings.CountTheNumberOfConsistentStrings_2(allowed,words);
    }

    public static int MinimumBitFlipsToConvertNumber(int start, int goal) {
        int xorResult = start ^ goal; // XOR gives the differing bits
        int count = 0;

        // Count the number of 1s in the XOR result
        while (xorResult > 0) {
            count += xorResult & 1; // Check the least significant bit
            xorResult >>= 1; // Shift right to check the next bit
        }

        return count;
    }
    //endregion

    //region Hashing
    public static int CountTheNumberOfConsistentStrings_2(String allowed, String[] words) {
        return CountTheNumberOfConsistentStrings_1(allowed,words);
    }

    public static boolean DivideArrayIntoEqualPairs(int[] nums) {
        return Array.DivideArrayIntoEqualPairs(nums);
    }

    public static String LongestNiceSubstring_1(String s) {
        return Strings.LongestNiceSubstring_2(s);
    }

    public static int MissingNumber(int[] numbers) {
        return HashTables.MissingNumber(numbers);
    }

    public static char FindTheDifference(String s, String t) {
        return Strings.FindTheDifference_2(s,t);
    }
    //endregion

    //region Back Tracking
    public static int CountNumberOfMaximumBitwiseORSubsets(int[] nums) {
        int maxOr = 0, count = 0;

        // Calculate the maximum bitwise OR of the entire array
        for (int num : nums) {
            maxOr |= num;
        }

        // Helper function for backtracking
        count = backtrack(nums, 0, 0, maxOr);
        return count;
    }

    public static List<String> LetterCasePermutation(String s) {
        return Strings.LetterCasePermutation_2(s);
    }

    public static List<List<Integer>> Subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(nums, 0, new ArrayList<>(), result);
        return result;
    }

    public static List<Integer> CircularPermutationInBinaryRepresentation(int n, int start) {
        List<Integer> grayCode = new ArrayList<>();

        // Generate Gray code for 2^n numbers
        for (int i = 0; i < (1 << n); i++) {
            grayCode.add(i ^ (i >> 1)); // Gray code formula
        }

        // Find the start index
        int startIndex = grayCode.indexOf(start);

        // Rearrange the sequence to start from the given start
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < grayCode.size(); i++) {
            result.add(grayCode.get((startIndex + i) % grayCode.size()));
        }

        return result;
    }

    public static int BeautifulArrangement_1(int n) {
        return Array.BeautifulArrangement(n);
    }
    //endregion

    //region DP Based
    public static int[] CountingBits(int n) {
        int[] ans = new int[n + 1]; // Initialize the result array

        // Loop to calculate the number of 1's for each number from 0 to n
        for (int i = 1; i <= n; i++) {
            ans[i] = ans[i >> 1] + (i & 1); // Using the relation described above
        }

        return ans;
    }

    public static int NumberOfGoodWaysToSplitAString(String s){
        return Strings.NumberOfGoodWaysToSplitAString(s);
    }

    public static int BeautifulArrangement_2(int n) {
        return BeautifulArrangement_1(n);
    }

    public static int MaximumCompatibilityScoreSum(int[][] students, int[][] mentors) {
        int m = students.length;
        int n = students[0].length;

        // Calculate compatibility scores between each student and mentor
        int[][] compatibility = new int[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                compatibility[i][j] = calculateCompatibility(students[i], mentors[j]);
            }
        }

        // DP array to store the maximum compatibility score for each bitmask
        int[] dp = new int[1 << m];

        // Iterate over all possible bitmasks
        for (int mask = 0; mask < (1 << m); mask++) {
            int numAssigned = Integer.bitCount(mask);  // Count how many mentors are assigned
            if (numAssigned == m) continue; // All mentors are assigned, skip

            // Try assigning a new student to the next unassigned mentor
            for (int mentor = 0; mentor < m; mentor++) {
                if ((mask & (1 << mentor)) == 0) {  // Mentor is unassigned in this mask
                    int newMask = mask | (1 << mentor);
                    dp[newMask] = Math.max(dp[newMask], dp[mask] + compatibility[numAssigned][mentor]);
                }
            }
        }

        // The answer will be stored in dp[(1 << m) - 1], which represents all mentors being assigned
        return dp[(1 << m) - 1];
    }

    public static int MaximumProductOfTheLengthOfTwoPalindromicSubsequences(String s) {
        int n = s.length();

        // Step 1: Generate all subsequences and check if they are palindromes.
        List<Integer> palindromicSubsequences = new ArrayList<>();

        // Check all possible subsequences using bitmasking
        for (int mask = 1; mask < (1 << n); mask++) {
            StringBuilder subsequence = new StringBuilder();
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    subsequence.append(s.charAt(i));
                }
            }

            // Check if the subsequence is a palindrome
            if (isPalindrome(subsequence.toString())) {
                palindromicSubsequences.add(mask);
            }
        }

        // Step 2: Find the two disjoint palindromic subsequences that maximize the product of their lengths
        int maxProduct = 0;

        // Try all pairs of disjoint subsequences
        for (int i = 0; i < palindromicSubsequences.size(); i++) {
            for (int j = i + 1; j < palindromicSubsequences.size(); j++) {
                int mask1 = palindromicSubsequences.get(i);
                int mask2 = palindromicSubsequences.get(j);

                // Check if the two subsequences are disjoint (no common bits)
                if ((mask1 & mask2) == 0) {
                    // Calculate the product of their lengths
                    int product = Integer.bitCount(mask1) * Integer.bitCount(mask2);
                    maxProduct = Math.max(maxProduct, product);
                }
            }
        }

        return maxProduct;
    }
    //endregion

    //region Sliding Window
    public static String LongestNiceSubstring(String s) {
        return LongestNiceSubstring_1(s);
    }

    public static List<String> RepeatedDnaSequences(String s) {
        List<String> result = new ArrayList<>();
        if (s.length() < 10) return result;  // If length is less than 10, no such subsequence exists.

        Set<String> seen = new HashSet<>();
        Set<String> repeated = new HashSet<>();

        for (int i = 0; i <= s.length() - 10; i++) {
            String substring = s.substring(i, i + 10);
            if (!seen.add(substring)) {  // If the substring was already seen, add it to repeated set
                repeated.add(substring);
            }
        }

        // Convert repeated set to result list
        result.addAll(repeated);

        return result;
    }
    //endregion

    //region Recursion
    public static boolean PowerOfTwo(int n) {
        if (n <= 0) {
            return false;  // Powers of two are positive numbers
        }
        return (n & (n - 1)) == 0;  // Check if n is a power of two
    }

    public static boolean PowerOfFour(int n) {
        if (n <= 0) {
            return false;  // Powers of four must be positive
        }
        // Check if n is a power of two and if the only 1 bit is at an even position
        return (n & (n - 1)) == 0 && (n - 1) % 3 == 0;
    }

    public static int[] MaximumPointsInAnArcheryCompetition(int numArrows, int[] aliceArrows) {
        int[] bob = new int[12];
        calculate(aliceArrows, bob, 11, numArrows, 0);  //Start with max point that is 11
        return maxbob;
    }
    //endregion

    //region Private Methods
    private static int backtrack(int[] nums, int index, int currentOr, int maxOr) {
        if (index == nums.length) {
            return currentOr == maxOr ? 1 : 0;
        }

        // Include the current element
        int include = backtrack(nums, index + 1, currentOr | nums[index], maxOr);

        // Exclude the current element
        int exclude = backtrack(nums, index + 1, currentOr, maxOr);

        return include + exclude;
    }

    private static void backtrack(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        // Add the current subset to the result
        result.add(new ArrayList<>(current));

        for (int i = start; i < nums.length; i++) {
            // Include nums[i] in the current subset
            current.add(nums[i]);

            // Recurse with the next element
            backtrack(nums, i + 1, current, result);

            // Backtrack by removing the last element
            current.remove(current.size() - 1);
        }
    }

    private static int calculateCompatibility(int[] student, int[] mentor) {
        int score = 0;
        for (int i = 0; i < student.length; i++) {
            if (student[i] == mentor[i]) {
                score++;
            }
        }
        return score;
    }

    private static boolean isPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    public static void calculate(int[] alice, int[] bob, int index, int remainArr, int point) {
        if(index < 0 || remainArr <= 0) {
            if(remainArr > 0)
                bob[0] += remainArr;
            if(point > bobPoint) { // Update the max points and result output
                bobPoint = point;
                maxbob = bob.clone();
            }
            return;
        }
        //part 1: assign 1 more arrow than alice
        if(remainArr >= alice[index]+1) {
            bob[index] = alice[index] + 1;
            calculate(alice, bob, index-1, remainArr-(alice[index]+1), point + index);
            bob[index] = 0;
        }
        //part 2: assign no arrow and move to next point
        calculate(alice, bob, index-1, remainArr, point);
        bob[index] = 0;
    }
    //endregion
}
