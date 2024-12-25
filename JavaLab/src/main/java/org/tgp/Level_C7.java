package org.tgp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Level_C7 {
    public static List<List<Integer>> CombinationSum1(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums); // Sort the array to handle elements in non-decreasing order
        backtrack(nums, target, 0, new ArrayList<>(), result);
        return result;
    }

    public static List<List<Integer>> CombinationSum2(int[] arr, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(arr); // Sort the array to handle elements in non-decreasing order
        backtrack(arr, target, 0, new ArrayList<>(), result);
        return result;
    }

    public static int BitConversion(int start, int goal) {
        int diff = start ^ goal;

        // Count the number of 1s in the binary representation of diff
        int flips = 0;
        while (diff != 0) {
            flips += (diff & 1); // Add 1 if the least significant bit is 1
            diff >>= 1; // Right shift the number by 1 to check the next bit
        }

        return flips;
    }

    //region Private Methods
    private static void backtrack(int[] arr, int target, int startIndex, List<Integer> currentCombination, List<List<Integer>> result) {
        // Base case: if the target is reached, add the current combination to the result
        if (target == 0) {
            result.add(new ArrayList<>(currentCombination));
            return;
        }

        // Explore the array starting from the current startIndex
        for (int i = startIndex; i < arr.length; i++) {
            // If the number exceeds the remaining target, no need to explore further
            if (arr[i] > target) {
                break;
            }

            // Include the number arr[i] and explore further with the reduced target
            currentCombination.add(arr[i]);
            backtrack(arr, target - arr[i], i, currentCombination, result); // We pass 'i' to allow repeated numbers
            currentCombination.remove(currentCombination.size() - 1); // Backtrack: remove the last element
        }
    }
    //endregion
}
