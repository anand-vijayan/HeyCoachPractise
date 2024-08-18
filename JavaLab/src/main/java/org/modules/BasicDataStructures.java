package org.modules;

import java.util.Arrays;

import static org.helpers.Searching.*;
import static org.helpers.Sorting.*;

public class BasicDataStructures {
    public static int MaximumSweetnessOfToffeeJar(int n, int[] price, int k) {

        //1. Sort the input array
        price = BubbleSort(price);

        //2. Define the search space
        int low = 0;
        int high = price[n - 1] - price[0];
        int result = 0;

        //3. Binary search on the minimum difference
        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (CanFormSweetness(price, n, k, mid)) {
                result = mid;  // mid is feasible, try for a larger difference
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        return result;
    }

    public static int SmallestDivisorSatisfyingTheLimit(int[] nums, int limit) {
        int low = 1;
        int high = Arrays.stream(nums).max().getAsInt();

        while (low < high) {
            int mid = low + (high - low) / 2;
            if (ComputeSum(nums, mid) <= limit) {
                high = mid; // Try a smaller divisor
            } else {
                low = mid + 1; // Increase the divisor
            }
        }

        return low;
    }

    public static int SearchInRotatedSortedArray(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        // Find the pivot (smallest element)
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        int pivot = left;
        left = 0;
        right = nums.length - 1;

        // Determine which part of the array to search
        if (target >= nums[pivot] && target <= nums[right]) {
            left = pivot;
        } else {
            right = pivot - 1;
        }

        // Binary search in the determined range
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1; // Target not found
    }

    public static int MedianOfTwoSortedArray(int[] arr1, int[] arr2) {
        //1. Sort both arrays
        arr1 = BubbleSort(arr1);
        arr2 = BubbleSort(arr2);

        //2. Merge them along with sorting
        int[] mergedArr = MergeSort(arr1,arr2);

        //3. Find middle
        int mid = mergedArr.length/2;
        int reminder = mergedArr.length%2;

        if(reminder == 0){
            return (int) Math.floor((mergedArr[mid-1] + mergedArr[mid]) / 2);
        } else {
            return mergedArr[mid];
        }
    }

    public static int SquareRootOf(int number) {
        int calc = 0, previousVal = 0;
        for(int i = 1; i <= number/2; i++) {
            calc = i * i;

            if(calc == number) return i;
            if(calc < number) previousVal = i;
            if(calc > number) return previousVal;
        }
        return previousVal;
    }


    private static int ComputeSum(int[] nums, int divisor) {
        int sum = 0;
        for (int num : nums) {
            sum += (int) Math.ceil((double) num / divisor);
        }
        return sum;
    }

    private static boolean CanFormSweetness(int[] price, int n, int k, int minDiff) {
        int count = 1;
        int lastSelected = price[0];

        for (int i = 1; i < n; i++) {
            if (price[i] - lastSelected >= minDiff) {
                count++;
                lastSelected = price[i];
            }
            if (count == k) {
                return true;
            }
        }
        return false;
    }
}
