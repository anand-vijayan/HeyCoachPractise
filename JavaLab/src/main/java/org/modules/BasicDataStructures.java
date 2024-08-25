package org.modules;

import java.util.*;

import static org.helpers.Common.*;
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

    public static int[] ReverseArray(int n, int[] a) {
        int[] b = new int[n*2];
        for(int i = 0; i < n; i++) {
            b[i] = b[(n*2)-i-1] = a[i];
        }
        return b;
    }

    public static int[] SoloSum(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        for(int i = 0; i < n; i++) {
            int leftSum = 0;
            int rightSum = 0;
            for(int j = i+1; j < n; j++) {
                rightSum += nums[j];
            }
            for(int k = i-1; k >= 0; k--) {
                leftSum += nums[k];
            }
            result[i] = leftSum + rightSum;
        }
        return result;
    }

    public static void CanYouMoveTwos(List<Integer> nums) {
        int[] arr = nums.stream().mapToInt(i->i).toArray();
        int n = arr.length;
        int lastNonTwoIndex = 0;
        for(int i = 0; i < n; i++) {
            if(arr[i] != 2) {
                arr[lastNonTwoIndex] = arr[i];
                lastNonTwoIndex++;
            }
        }

        for(int i = lastNonTwoIndex; i < n; i++) {
            arr[i] = 2;
        }

        PrintArray(arr);
    }

    public static int FindTheXorConnects(ArrayList<Integer> a, ArrayList<Integer> b){
        int result = 0;
        for(int i : a) {
            for(int j : b) {
                if((i ^ j) == 0) result++;
            }
        }
        return result;
    }

    public static void MrYadavsVacation(int[] weather, int n, int k, int t) {
        int validDays = 0, count = 0;
        //count valid maximum consecutive days
        for(int i = 0; i < n; i++) {
            if(weather[i] <= t) {
                validDays++;
            } else {
                count += CountDays(validDays, k);
                validDays = 0;
            }
        }

        count += CountDays(validDays, k);

        System.out.println(count);
    }

    public static int LongestSubstringWithKUniqueChar(String s, int k) {

        if(s == null || s.isEmpty() || k <= 0){
            return -1;
        }

        int n = s.length();
        int maxLength = -1;
        int left = 0;

        HashMap<Character, Integer> charCountMap = new HashMap<>();

        for(int right = 0; right < n; right++) {
            char rightChar = s.charAt(right);
            charCountMap.put(rightChar, charCountMap.getOrDefault(rightChar,0) + 1);

            while (charCountMap.size() > k) {
                char leftChar = s.charAt(left);
                charCountMap.put(leftChar, charCountMap.get(leftChar) - 1);
                if(charCountMap.get(leftChar) == 0) {
                    charCountMap.remove(leftChar);
                }
                left++;
            }

            if(charCountMap.size() == k) {
                maxLength = Math.max(maxLength, right - left + 1);
            }
        }

        return maxLength;
    }

    public static int LongestSubstringWithNoRepeatingChar(String str) {

        if(str == null || str.isEmpty()) {
            return -1;
        }

        HashSet<Character> uniqueChars = new HashSet<>();
        int i = 0, left = 0, n = str.length(), maxLength = 0;

        for(int right = 0; right < n; right++) {
            char rightChar = str.charAt(right);

            while (uniqueChars.contains(rightChar)) {
                uniqueChars.remove(str.charAt(left));
                left++;
            }

            uniqueChars.add(rightChar);
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }

    public static List<Integer> UnravelingCluesWithProfessor(String text, String pattern) {
        List<Integer> result = new ArrayList<>();
        int n = text.length();
        int m = pattern.length();

        if (m == 0 || n < m) {
            return result;  // If pattern is empty or larger than text, return empty list
        }

        // Create the LPS (Longest Prefix Suffix) array
        int[] lps = createLPSArray(pattern);

        int i = 0; // Index for text
        int j = 0; // Index for pattern

        while (i < n) {
            if (pattern.charAt(j) == text.charAt(i)) {
                i++;
                j++;
            }

            if (j == m) {
                result.add(i - j); // Pattern found, add the start index to the result
                j = lps[j - 1];    // Continue to search for more patterns
            } else if (i < n && pattern.charAt(j) != text.charAt(i)) {
                if (j != 0) {
                    j = lps[j - 1]; // Skip unnecessary comparisons
                } else {
                    i++;
                }
            }
        }

        return result;
    }

    public static String MinWindowSubstring(String s, String t) {
        if (s == null || t == null || s.length() < t.length()) {
            return "";
        }

        // Step 1: Create a frequency map for string t
        Map<Character, Integer> tFreq = new HashMap<>();
        for (char c : t.toCharArray()) {
            tFreq.put(c, tFreq.getOrDefault(c, 0) + 1);
        }

        // Step 2: Use a sliding window to find the minimum window
        int left = 0, right = 0;
        int minLength = Integer.MAX_VALUE;
        int minLeft = 0;
        int count = 0; // Number of characters matched
        Map<Character, Integer> windowFreq = new HashMap<>();

        while (right < s.length()) {
            char rightChar = s.charAt(right);
            if (tFreq.containsKey(rightChar)) {
                windowFreq.put(rightChar, windowFreq.getOrDefault(rightChar, 0) + 1);
                if (windowFreq.get(rightChar).intValue() <= tFreq.get(rightChar).intValue()) {
                    count++;
                }
            }

            // Step 3: Shrink the window from the left to find the minimum size
            while (count == t.length()) {
                if (right - left + 1 < minLength) {
                    minLength = right - left + 1;
                    minLeft = left;
                }

                char leftChar = s.charAt(left);
                if (tFreq.containsKey(leftChar)) {
                    windowFreq.put(leftChar, windowFreq.get(leftChar) - 1);
                    if (windowFreq.get(leftChar) < tFreq.get(leftChar)) {
                        count--;
                    }
                }
                left++;
            }

            right++;
        }

        // Step 4: Return the result
        return minLength == Integer.MAX_VALUE ? "" : s.substring(minLeft, minLeft + minLength);
    }

    public static String MarsMission(int n, String word) {
        List<String> syllables = new ArrayList<>();
        List<Character> temp = new ArrayList<>();
        int k = 0;

        while(k <= n) {
            char c = (k == n) ? Character.MIN_VALUE : word.charAt(k);
            if(c != Character.MIN_VALUE && temp.size() < 3) {
                temp.add(c);
            } else if(temp.size() == 3) {
                if(isVowel(c)) {
                    String type1Str = "";
                    int i = 0;
                    while(i < 2) {
                        type1Str += temp.get(i);
                        i++;
                    }
                    char tempCarryForward = temp.get(i);
                    temp.clear();
                    temp.add(tempCarryForward);
                    syllables.add(type1Str);
                } else if(c != Character.MIN_VALUE && !isVowel(c)) {
                    String type2Str = "";
                    int j = 0;
                    while(j < 3) {
                        type2Str += temp.get(j);
                        j++;
                    }
                    temp.clear();
                    syllables.add(type2Str);
                } else {
                    String tempLeftOver = "";
                    int l = 0;
                    while(l < temp.size()) {
                        tempLeftOver += temp.get(l);
                        l++;
                    }
                    temp.clear();
                    syllables.add(tempLeftOver);
                }
                temp.add(c);
            } else {
                String tempLeftOver = "";
                int l = 0;
                while(l < temp.size()) {
                    tempLeftOver += temp.get(l);
                    l++;
                }
                temp.clear();
                syllables.add(tempLeftOver);
            }
            k++;
        }

        return String.join(" ", syllables);
    }

    public static void PrintFirstNegativeNumberInEveryWindowOfSizeK(ArrayList<Integer> arr, int k){
        List<Integer> result = new ArrayList<>();
        int left = 0, right = k - 1;
        boolean negFound;

        while(right < arr.size()) {
            negFound = false;
            for(int i = left; i <= right; i++) {
                if(arr.get(i) < 0) {
                    result.add(arr.get(i));
                    negFound = true;
                    break;
                }
            }
            left++;
            right = left + k - 1;
            if(!negFound)
                result.add(0);
        }

        PrintArray(result);
    }

    public static int CountAnagrams(String s, String c) {
        int count = 0;
        int n = s.length();
        int k = c.length();

        //If pattern longer than text, then no Anagrams possible
        if (k > n) {
            return 0;
        }

        // Frequency map for characters in pattern.
        Map<Character, Integer> cMap = new HashMap<>();
        for (char ch : c.toCharArray()) {
            cMap.put(ch, cMap.getOrDefault(ch, 0) + 1);
        }

        // Frequency map for the current window in text.
        Map<Character, Integer> sMap = new HashMap<>();
        for (int i = 0; i < k; i++) {
            char ch = s.charAt(i);
            sMap.put(ch, sMap.getOrDefault(ch, 0) + 1);
        }

        // Compare the first window.
        if (sMap.equals(cMap)) {
            count++;
        }

        // Slide the window over S.
        for (int i = k; i < n; i++) {
            // Remove the first character of the previous window.
            char toRemove = s.charAt(i - k);
            sMap.put(toRemove, sMap.get(toRemove) - 1);
            if (sMap.get(toRemove) == 0) {
                sMap.remove(toRemove);
            }

            // Add the new character to the window.
            char toAdd = s.charAt(i);
            sMap.put(toAdd, sMap.getOrDefault(toAdd, 0) + 1);

            // Compare maps again.
            if (sMap.equals(cMap)) {
                count++;
            }
        }

        return count;
    }

    public static int LongestWorkingLights(String s, int k) {
        int start = 0;
        int maxLength = 0;
        int currentDamaged = 0;
        int n = s.length();

        for (int end = 0; end < n; end++) {
            if (s.charAt(end) == '.') {
                currentDamaged++;
            }

            while (currentDamaged > k) {
                if (s.charAt(start) == '.') {
                    currentDamaged--;
                }
                start++;
            }

            maxLength = Math.max(maxLength, end - start + 1);
        }

        return maxLength;
    }

    public static boolean RohanLovesZero1(int[] arr, int N) {
        int left = 0, sum = 0;
        while(left < N) {
            for(int i = left; i < N; i++) {
                sum = 0;
                for(int j = left; j <= i; j++) {
                    sum += arr[j];
                }

                if(sum == 0) return true;
            }
            left++;
        }
        return false;
    }

    public static List<Integer> RohanLovesZero2(List<Integer> arr) {
        List<Integer> result = new ArrayList<>();
        int left, right, rSum, lSum;

        for(int i = 0; i < arr.size(); i++) {
            left = 0;
            lSum = 0;
            rSum = 0;
            right = arr.size() - 1;

            while(left < i) {
                lSum += arr.get(left);
                left++;
            }

            while (right > i) {
                rSum += arr.get(right);
                right--;
            }

            if(lSum == rSum){
                result.add(i);
            }
        }

        return result;
    }

    public static int MinAbsoluteDifference(List<Integer> nums, int x) {
        int minDiff = Integer.MAX_VALUE;
        TreeSet<Integer> setValues = new TreeSet<>();
        int n = nums.size();

        for(int i = 0; i < n; i++) {
            if(i >= x) {
                int num = nums.get(i);

                //Check lower bound
                Integer lower = setValues.floor(num);
                if(lower != null) {
                    minDiff = Math.min(minDiff, Math.abs(num - lower));
                }

                //Check upper bound
                Integer upper = setValues.ceiling(num);
                if(upper != null) {
                    minDiff = Math.min(minDiff, Math.abs(num - upper));
                }

                setValues.remove(nums.get(i - x));
            }

            setValues.add(nums.get(i));
        }

        return minDiff;
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

    private static int CountDays(int maxRange, int allowedRange) {
        if (maxRange < allowedRange) {
            return 0;
        }

        return (maxRange - allowedRange + 1) * (maxRange - allowedRange + 2) / 2;
    }

    private static int[] createLPSArray(String pattern) {
        int m = pattern.length();
        int[] lps = new int[m];
        int length = 0; // Length of the previous longest prefix suffix
        int i = 1;

        lps[0] = 0; // LPS of the first character is always 0

        while (i < m) {
            if (pattern.charAt(i) == pattern.charAt(length)) {
                length++;
                lps[i] = length;
                i++;
            } else {
                if (length != 0) {
                    length = lps[length - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }

        return lps;
    }

    private static boolean isVowel(char c) {
        return "aeiou".indexOf(c) != -1;
    }
}
