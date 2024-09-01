package org.helpers;

public class Searching {
    public static int BinarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

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
}
