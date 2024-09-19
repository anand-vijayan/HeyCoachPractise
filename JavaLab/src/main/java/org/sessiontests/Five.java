package org.sessiontests;

import java.util.*;

public class Five {

    //region Variables & Constants
    private static final int MOD = 1_000_000_007;
    private static int n;
    private static List<Integer> gems;
    private static int validCount;
    //endregion

    public static int Tribonacci(int n) {
        // Handle base cases
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        // Initialize the first three numbers of the Tribonacci sequence
        int[] tribonacci = new int[n + 1];
        tribonacci[0] = 0;
        tribonacci[1] = 1;
        tribonacci[2] = 1;

        // Compute the nth Tribonacci number iteratively
        for (int i = 3; i <= n; i++) {
            tribonacci[i] = tribonacci[i - 1] + tribonacci[i - 2] + tribonacci[i - 3];
        }

        // Return the nth Tribonacci number
        return tribonacci[n];
    }

    public static int MaximizePoints(int[] nums) {
        if (nums == null || nums.length == 0) return 0;

        // Step 1: Calculate frequency of each gem value
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        int maxVal = 0;
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
            maxVal = Math.max(maxVal, num);
        }

        // Step 2: Initialize DP array
        int[] dp = new int[maxVal + 1];

        // Base cases
        dp[0] = 0; // If value 0 exists (not used in this problem)
        dp[1] = frequencyMap.getOrDefault(1, 0) * 1;

        // Step 3: Fill the DP array
        for (int i = 2; i <= maxVal; i++) {
            int pointsIfTake = i * frequencyMap.getOrDefault(i, 0);
            dp[i] = Math.max(dp[i-1], pointsIfTake + dp[i-2]);
        }

        // The answer is the maximum points considering all values up to maxVal
        return dp[maxVal];
    }

    public static int SpecialArrangements(List<Integer> numbers) {
        int[] nums = numbers.stream().mapToInt(i -> i).toArray();
        return countSpecialArrangements(nums);
    }

    //region Private Methods
    private static int countSpecialArrangements(int[] nums) {
        n = nums.length;
        gems = new ArrayList<>();
        for (int num : nums) {
            gems.add(num);
        }
        validCount = 0;

        // Generate permutations and count valid ones
        generatePermutations(new ArrayList<>(), new boolean[n]);

        return validCount;
    }

    private static void generatePermutations(List<Integer> currentPermutation, boolean[] used) {
        if (currentPermutation.size() == n) {
            if (isValidPermutation(currentPermutation)) {
                validCount = (validCount + 1) % MOD;
            }
            return;
        }

        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            used[i] = true;
            currentPermutation.add(gems.get(i));
            generatePermutations(currentPermutation, used);
            currentPermutation.remove(currentPermutation.size() - 1);
            used[i] = false;
        }
    }

    private static boolean isValidPermutation(List<Integer> permutation) {
        for (int i = 0; i < n - 1; i++) {
            int a = permutation.get(i);
            int b = permutation.get(i + 1);
            if (!(a % b == 0 || b % a == 0)) {
                return false;
            }
        }
        return true;
    }
    //endregion

}
