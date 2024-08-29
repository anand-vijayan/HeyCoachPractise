package org.modules;

import org.dto.Node;
import org.dto.Pair;
import org.dto.PetrolPump;

import java.util.*;

import static org.helpers.Common.*;
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
        int[] lps = CreateLPSArray(pattern);

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
                if(IsVowel(c)) {
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
                } else if(c != Character.MIN_VALUE && !IsVowel(c)) {
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

        for (int i = 0; i < nums.size(); i++) {
            if(i + x < nums.size()) {
                minDiff = Math.min(minDiff, Math.abs(nums.get(i) - nums.get(i + x)));
            }
        }

        return minDiff;
    }

    public static ArrayList<Integer> NextLeastGreatest(int n, int[] arr) {
        ArrayList<Integer> result = new ArrayList<>();
        TreeSet<Integer> nextLeastGreatest = new TreeSet<>();
        for(int i = 0; i < n; i ++) {
            for(int j = i + 1; j < n; j++) {
                if(arr[j] > arr[i]) {
                    nextLeastGreatest.add(arr[j]);
                }
            }
            //Now check if there are greatest of arr[i] in the set, if found get minimum value from set and enter for arr[i] else enter -1
            if(nextLeastGreatest.isEmpty()) {
                result.add(-1);
            } else {
                result.add(nextLeastGreatest.first());
                nextLeastGreatest.clear();
            }
        }
        return result;
    }

    public static void UnionAndIntersection(int[] arr1, int[] arr2, int n, int m) {
        TreeSet<Integer> arr1Set = new TreeSet<>();
        TreeSet<Integer> arr2Set = new TreeSet<>();
        n = arr1.length;
        m = arr2.length;

        //Get all value from arr1, TreeSet will not take duplicate value
        for(int i = 0; i < n; i++) {
            arr1Set.add(arr1[i]);
        }

        //Get all value from arr2, TreeSet will not take duplicate value
        for(int i = 0; i < m; i++) {
            arr2Set.add(arr2[i]);
        }

        //Now, union both
        TreeSet<Integer> unionSet = new TreeSet<>(arr1Set);
        unionSet.addAll(arr2Set);

        //Then find intersection
        TreeSet<Integer> intersectionSet = new TreeSet<>(arr1Set);
        intersectionSet.retainAll(arr2Set);

        //Print answers
        PrintArray(Arrays.asList(unionSet.toArray()));
        PrintArray(Arrays.asList(intersectionSet.toArray()));
    }

    public static void UniquePermutations(String s) {
        Set<String> permutations = new TreeSet<>();

        // Generate permutations
        GeneratePermutations(s, "", permutations);

        // Print each permutation
        for (String permutation : permutations) {
            System.out.println(permutation);
        }
    }

    public static void JetFighterCaptain(int[] a, int n, int k) {
        //Find number of segments, if it is 1 or any odd then it is not valid message so "Wait". Else continue.
        int numberOfSegments = n/k, start = 0, missingPositiveNumbers = 0;
        boolean areSegmentsSame = false;
        List<Set<Integer>> segments = new ArrayList<>();
        if(numberOfSegments == 1) {
            System.out.println("Wait");
        } else{
            //Create segments of equal length
            for(int i = 1; i <= numberOfSegments; i++) {
                segments.add(new TreeSet<>(Arrays.asList(GetSegment(a,start,k))));
                start += k;
            }

            //Check if all sets are same
            for(int i = 0; i < segments.size() - 1; i++) {
                if(!segments.get(i).containsAll(segments.get(i+1))) {
                    areSegmentsSame = false;
                    break;
                } else {
                    areSegmentsSame = true;
                }
            }

            if(areSegmentsSame) {
                //Check if any positive number is missing in the segments
                for(Set<Integer> segment : segments) {
                    missingPositiveNumbers += CheckMissingPositiveNumber(segment);
                }
                //If number of segments and missing positive numbers are same then "Attack" else "Wait"
                System.out.println((numberOfSegments == missingPositiveNumbers) ? "Attack" : "Wait");
            } else {
                System.out.println("Wait");
            }
        }
    }

    public static int MostFrequentElement(int[] arr) {
        int highestFrequency = 0, highestRepeatedElement = Integer.MAX_VALUE;
        Map<Integer, Integer> elementFrequency = new HashMap<>();

        for(int i : arr) {
            elementFrequency.put(i, elementFrequency.getOrDefault(i, 0) + 1);
        }

        for(Map.Entry<Integer, Integer> element : elementFrequency.entrySet()) {
            int elementVal = element.getKey();
            int elementCount = element.getValue();

            if(elementCount > highestFrequency ||
                    elementCount == highestFrequency && elementVal < highestRepeatedElement) {
                highestFrequency = elementCount;
                highestRepeatedElement = elementVal;
            }
        }

        return highestRepeatedElement;
    }

    public static Map<Integer, Integer> HashMapImplementation(int[] arr, int n) {
        Map<Integer, Integer> result = new HashMap<>();
        //Since hashMap store data as sorted (ascending), we don't need a sort operation
        //arr = BubbleSort(arr);
        for(int i : arr) {
            result.put(i, result.getOrDefault(i, 0) + 1);
        }
        return result;
    }

    public static String BerloggingWinner(List<Pair<String, Integer>> input, int N) {
        int highestScore = Integer.MIN_VALUE;
        String winner = "";
        Map<String, Integer> scoreSheet = new HashMap<>();
        for(Pair<String, Integer> candidateScore : input) {
            scoreSheet.put(candidateScore.getKey(), scoreSheet.getOrDefault(candidateScore.getKey(), 0) + candidateScore.getValue());
        }

        for(Map.Entry<String, Integer> score : scoreSheet.entrySet()) {
            int candidateFinalScore = score.getValue();
            if(candidateFinalScore > highestScore) {
                highestScore = candidateFinalScore;
                winner = score.getKey();
            }
        }

        //Check if there is more than one candidate with same score
        int countOfHighestScore = 0;
        for(int val : scoreSheet.values()) {
            if(val == highestScore) countOfHighestScore++;
        }
        //There is only one candidate, and we captured it in 'winner', so return it.
        if(countOfHighestScore == 1) {
            return winner;
        } else {
            //There are more than one with same score.
            //Find the first person got into that score
            Map<String, Integer> duplicateScoreSheet = new HashMap<>();
            for(Pair<String, Integer> candidateScore : input) {
                duplicateScoreSheet.put(
                        candidateScore.getKey(),
                        duplicateScoreSheet.getOrDefault(candidateScore.getKey(), 0) + candidateScore.getValue());

                if(duplicateScoreSheet.get(candidateScore.getKey()) == highestScore) {
                    return candidateScore.getKey();
                }
            }
        }

        return winner;
    }

    public static int NumberOfSubArraysWithGivenSum(int[] arr, int n, int sum) {
        Map<Integer, Integer> sumFrequencyMap = new HashMap<>();
        int curr_sum = 0;
        int count = 0;

        // Initialize the map with sum 0 (important for subarrays starting from index 0)
        sumFrequencyMap.put(0, 1);

        for (int num : arr) {
            curr_sum += num;

            // Check if there is a subarray (ending at the current index) with sum k
            if (sumFrequencyMap.containsKey(curr_sum - sum)) {
                count += sumFrequencyMap.get(curr_sum - sum);
            }

            // Update the frequency of the current sum in the map
            sumFrequencyMap.put(curr_sum, sumFrequencyMap.getOrDefault(curr_sum, 0) + 1);
        }

        return count;
    }

    public static int MaxNumberOfHappiestGuests(int k, List<List<Integer>> nums, int n, int m) {
        int totalFruits = n * m, happyGuests = 0;

        //Get the fruit list
        Map<Integer, Integer> availableFruits = new HashMap<>();
        for (int i = 1; i <= totalFruits; i++) {
            availableFruits.put(i, 0);
        }

        //Assign the choices
        int guestId = 1;
        for(List<Integer> choice : nums) {
            do {
                int maxChoice = Collections.max(choice);
                if(availableFruits.get(maxChoice) != null) {
                    availableFruits.put(maxChoice, guestId);
                    break;
                }
                choice.remove(maxChoice);
            } while(!choice.isEmpty());
            guestId++;
        }

        for(Map.Entry<Integer, Integer> availableFruit : availableFruits.entrySet()) {
            if(availableFruit.getValue() > 0) {
                happyGuests++;
            }
        }

        return happyGuests;
    }

    public static int MinCost(int[] nums) {
        int n = nums.length;
        int minCost = Integer.MAX_VALUE;

        // Iterate over all possible ways to split the array into 3 subarrays
        for (int i = 1; i < n - 1; i++) {   // `i` is the end of the first subarray
            for (int j = i + 1; j < n; j++) { // `j` is the end of the second subarray
                int cost = nums[0] + nums[i] + nums[j];
                minCost = Math.min(minCost, cost);
            }
        }

        return minCost;
    }

    public static void SplitCircularLinkedList(Node head, Node[] result) {
        if (head == null || head.next == head) {
            result[0] = head;
            result[1] = null;
            return;
        }

        Node head1 = null, head2 = null;
        Node current1 = null, current2 = null;
        Node current = head;
        boolean addToFirstList = true;

        do {
            if (addToFirstList) {
                if (head1 == null) {
                    head1 = current;
                    current1 = head1;
                } else {
                    current1.next = current;
                    current1 = current1.next;
                }
            } else {
                if (head2 == null) {
                    head2 = current;
                    current2 = head2;
                } else {
                    current2.next = current;
                    current2 = current2.next;
                }
            }

            addToFirstList = !addToFirstList;
            current = current.next;
        } while (current != head);

        // Completing the circular nature
        current1.next = head1;
        current2.next = head2;

        result[0] = head1;
        result[1] = head2;
    }

    public static int MaximumGap(int[] arr){
        int n = arr.length;
        if (n < 2) {
            return 0;
        }

        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        // Find the minimum and maximum values in the array
        for (int num : arr) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }

        if (min == max) {
            return 0; // All elements are the same
        }

        // Calculate the bucket size and number of buckets
        int bucketSize = Math.max(1, (max - min) / (n - 1));
        int bucketCount = (max - min) / bucketSize + 1;

        // Create buckets
        int[] bucketMin = new int[bucketCount];
        int[] bucketMax = new int[bucketCount];
        Arrays.fill(bucketMin, Integer.MAX_VALUE);
        Arrays.fill(bucketMax, Integer.MIN_VALUE);

        // Place each number in a bucket
        for (int num : arr) {
            int bucketIndex = (num - min) / bucketSize;
            bucketMin[bucketIndex] = Math.min(bucketMin[bucketIndex], num);
            bucketMax[bucketIndex] = Math.max(bucketMax[bucketIndex], num);
        }

        // Find the maximum gap between successive buckets
        int maxGap = 0;
        int previousMax = bucketMax[0];

        for (int i = 1; i < bucketCount; i++) {
            if (bucketMin[i] == Integer.MAX_VALUE) {
                // Skip empty buckets
                continue;
            }
            maxGap = Math.max(maxGap, bucketMin[i] - previousMax);
            previousMax = bucketMax[i];
        }

        return maxGap;
    }

    public static Queue<Integer> QueueImplementation(Vector<Integer> nums) {
        Queue<Integer> queue = new LinkedList<>();
        for (int num : nums) {
            queue.add(num);
        }
        return queue;
    }

    public static Queue<Integer> QueueFunctions(Vector<Integer> v, int k) {
        Queue<Integer> dummyQueue = new LinkedList<>();

        //1. Implement a queue with integers from the given array.
        for(int num : v) {
            dummyQueue.add(num);
        }

        //2: Remove first element from the queue
        dummyQueue.remove();

        //3. Print the first element of queue.
        System.out.println(dummyQueue.element());

        //4.Find if the integer k exists in the queue.
        System.out.println(dummyQueue.contains(k) ? "Yes" : "No");

        //5. Finally return the queue.
        return dummyQueue;
    }

    public static Queue<Integer> ReverseKQueueElements(Queue<Integer> q, int k) {
        if(q == null || q.isEmpty() || k > q.size() || k <= 0) {
            return q;
        }

        Stack<Integer> dummyStack = new Stack<>();

        //Push first 'k' elements to a stack
        for(int i = 0; i < k; i++) {
            dummyStack.push(q.poll());
        }

        //Add it back to queue from stack
        while (!dummyStack.isEmpty()) {
            q.add(dummyStack.pop());
        }

        //Add the remaining elements in the queue (after 'k' elements)
        for(int i = 0; i < q.size() - k; i++) {
            q.add(q.poll());
        }

        return q;
    }

    public static int TourOfAllPetrolPump(PetrolPump[] p, int n) {
        int totalPetrol = 0;
        int totalDistance = 0;
        int start = 0;
        int currentPetrol = 0;

        for (int i = 0; i < p.length; i++) {
            totalPetrol += p[i].petrol;
            totalDistance += p[i].distance;
            currentPetrol += p[i].petrol - p[i].distance;

            // If current petrol in the truck becomes negative, the starting point can't be this or any pump before this.
            if (currentPetrol < 0) {
                start = i + 1;
                currentPetrol = 0;
            }
        }

        // If total petrol is less than total distance, then a tour is not possible.
        if (totalPetrol < totalDistance) {
            return -1;
        }

        return start;
    }

    public static String FirstNonRepeating(String str) {
        StringBuilder outputStr = new StringBuilder();
        HashMap<Character, Integer> freqMap = new HashMap<>();
        Queue<Character> queue = new LinkedList<>();

        for (char ch : str.toCharArray()) {
            // Add character to the queue and update its frequency
            queue.add(ch);
            freqMap.put(ch, freqMap.getOrDefault(ch, 0) + 1);

            // Remove characters from the queue that are repeating
            while (!queue.isEmpty() && freqMap.get(queue.peek()) > 1) {
                queue.poll();
            }

            // If queue is not empty, append the first non-repeating character
            if (!queue.isEmpty()) {
                outputStr.append(queue.peek());
            } else {
                outputStr.append('X');
            }
        }

        return outputStr.toString();
    }

    public static boolean CalculateTotalTrappedWorkers(Vector<Integer> in, Vector<Integer> out) {
        Stack<Integer> stack = new Stack<>();
        int j = 0;

        for (Integer integer : in) {
            stack.push(integer);

            // While the stack is not empty and the top of the stack matches the OUT sequence
            while (!stack.isEmpty() && Objects.equals(stack.peek(), out.get(j))) {
                stack.pop();
                j++;
            }
        }

        // If stack is empty, it means all workers came out safely
        return !stack.isEmpty();
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

    private static int[] CreateLPSArray(String pattern) {
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

    private static boolean IsVowel(char c) {
        return "aeiou".indexOf(c) != -1;
    }

    private static void GeneratePermutations(String str, String prefix, Set<String> result) {
        if (str.isEmpty()) {
            result.add(prefix);
        } else {
            for (int i = 0; i < str.length(); i++) {
                String remaining = str.substring(0, i) + str.substring(i + 1);
                GeneratePermutations(remaining, prefix + str.charAt(i), result);
            }
        }
    }

    private static Integer[] GetSegment(int[] arr, int start, int len) {
        Integer[] result = new Integer[len];
        int j = 0;
        for(int i = start; i < start + len; i++) {
            result[j] = arr[i];
            j++;
        }
        return result;
    }

    private static int CheckMissingPositiveNumber(Set<Integer> segment) {
        //Check for missing number
        int missingNumber = 0;
        for (int num : segment) {
            if (num <= 0) {
                missingNumber++;
            }
        }

        return missingNumber;
    }
}
