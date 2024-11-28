package org.mindmaps;

import java.util.*;

public class SearchingAndSorting {

    //region Binary Search
    public static List<Integer> FindTargetIndicesAfterSortingArray(int[] nums, int target) {
        // Step 1: Sort the array
        Arrays.sort(nums);

        // Step 2: Find target indices
        List<Integer> result = new ArrayList<>();

        // Step 3: Iterate through the sorted array and collect indices of the target
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                result.add(i);
            }
        }

        return result;
    }

    public static int CountNegativeNumbersInASortedMatrix(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int count = 0;
        int row = 0;
        int col = n - 1;

        // Traverse the matrix starting from top-right corner
        while (row < m && col >= 0) {
            if (grid[row][col] < 0) {
                // All elements to the left of this element in the row are negative
                count += (m - row);
                col--;
            } else {
                row++;
            }
        }

        return count;
    }

    public static int[] TheKWeakestRowsInAMatrix(int[][] mat, int k) {
        int m = mat.length; // number of rows
        int n = mat[0].length; // number of columns
        int[] result = new int[k];

        // List to store (number of soldiers, row index)
        List<int[]> soldiersWithIndex = new ArrayList<>();

        // Step 1: Count the number of soldiers in each row
        for (int i = 0; i < m; i++) {
            int soldiers = countSoldiers(mat[i]);
            soldiersWithIndex.add(new int[]{soldiers, i});
        }

        // Step 2: Sort by the number of soldiers, and by row index if equal number of soldiers
        soldiersWithIndex.sort((a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

        // Step 3: Get the indices of the k weakest rows
        for (int i = 0; i < k; i++) {
            result[i] = soldiersWithIndex.get(i)[1];
        }

        return result;
    }

    public static int PeakIndexInAMountainArray(int[] arr) {
        int low = 0, high = arr.length - 1;

        while (low < high) {
            int mid = low + (high - low) / 2;

            // Check if we are on the decreasing part of the array
            if (arr[mid] < arr[mid + 1]) {
                low = mid + 1; // Peak must be on the right side
            } else {
                high = mid; // Peak is at mid or on the left side
            }
        }

        return low;
    }

    public static int[] IntersectionOfTwoArrays(int[] array1, int[] array2) {
        // Convert arrays to sets to remove duplicates and ensure uniqueness
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();

        for (int num : array1) {
            set1.add(num);  // Add elements from nums1 to set1
        }

        for (int num : array2) {
            set2.add(num);  // Add elements from nums2 to set2
        }

        // Find the intersection of the two sets
        set1.retainAll(set2);  // Retain only the elements that are in both sets

        // Convert the resulting set to an array and return it
        int[] result = new int[set1.size()];
        int index = 0;
        for (int num : set1) {
            result[index++] = num;
        }

        return result;
    }
    //endregion

    //region Sorting
    public static int MinimumSumOfFourDigitNumberAfterSplittingDigits(int num) {
        // Convert num to a string and then to a char array
        String numStr = Integer.toString(num);
        char[] digits = numStr.toCharArray();

        // Sort the digits in ascending order
        Arrays.sort(digits);

        // Two new integers as strings
        StringBuilder new1 = new StringBuilder();
        StringBuilder new2 = new StringBuilder();

        // Distribute digits alternately
        for (int i = 0; i < digits.length; i++) {
            if (i % 2 == 0) {
                new1.append(digits[i]);
            } else {
                new2.append(digits[i]);
            }
        }

        // Convert the two numbers from strings to integers
        int num1 = Integer.parseInt(new1.toString());
        int num2 = Integer.parseInt(new2.toString());

        // Return the sum of the two numbers
        return num1 + num2;
    }

    public static int[] HowManyNumbersAreSmallerThanTheCurrentNumber(int[] array) {
        // Step 1: Create a copy of nums and sort it
        int[] sortedNumbers = array.clone();
        Arrays.sort(sortedNumbers);

        // Step 2: Create a map to store the first occurrence index of each number
        Map<Integer, Integer> numToSmallerCount = new HashMap<>();

        for (int i = 0; i < sortedNumbers.length; i++) {
            // Only set the value for the first occurrence
            if (!numToSmallerCount.containsKey(sortedNumbers[i])) {
                numToSmallerCount.put(sortedNumbers[i], i);
            }
        }

        // Step 3: Construct the result array based on the map
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = numToSmallerCount.get(array[i]);
        }

        return result;
    }

    public static String SortingTheSentence(String s) {
        // Step 1: Split the shuffled sentence into words
        String[] words = s.split(" ");

        // Step 2: Sort the words based on the number at the end of each word
        Arrays.sort(words, (word1, word2) -> {
            // Compare the number at the end of each word
            return Integer.compare(word1.charAt(word1.length() - 1), word2.charAt(word2.length() - 1));
        });

        // Step 3: Remove the number from each word and reconstruct the original sentence
        StringBuilder result = new StringBuilder();
        for (String word : words) {
            // Remove the last character (the number)
            result.append(word, 0, word.length() - 1).append(" ");
        }

        // Remove the trailing space and return the result
        return result.toString().trim();
    }

    public static int MaximumProductDifferenceBetweenTwoPairs(int[] array) {
        // Step 1: Sort the array
        Arrays.sort(array);

        // Step 2: Calculate the product difference
        int n = array.length;
        int maxProduct = array[n - 1] * array[n - 2];  // Two largest numbers
        int minProduct = array[0] * array[1];          // Two smallest numbers

        // Step 3: Return the product difference
        return maxProduct - minProduct;
    }

    public static int MaximumProductOfTwoElementsInAnArray(int[] numbers) {
        // Initialize two variables to store the two largest numbers
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;

        // Traverse the array and find the two largest numbers
        for (int num : numbers) {
            if (num > max1) {
                max2 = max1;  // Update the second largest
                max1 = num;   // Update the largest
            } else if (num > max2) {
                max2 = num;   // Update the second largest
            }
        }

        // Return the maximum product of (max1-1) and (max2-1)
        return (max1 - 1) * (max2 - 1);
    }
    //endregion

    //region Private Methods

    private static int countSoldiers(int[] row) {
        int count = 0;
        for (int value : row) {
            if (value == 1) {
                count++;
            } else {
                break;
            }
        }
        return count;
    }

    //endregion
}
