package org.mindmaps;

import org.dto.ListNode;
import org.dto.TreeNode;
import org.helpers.FindElements;
import org.helpers.ThroneInheritance;

import java.util.*;
import java.util.function.Supplier;

public class HashTables {
    //region Variables & Constants
    private static int maxDepth = -1;
    private static TreeNode lcaNode = null;
    private static final TreeNode result = null;
    //endregion

    //region Simple Hashing
    public static int JewelsAndStones(String jewels, String stones) {
        // Convert jewels to a HashSet for O(1) membership checking
        HashSet<Character> jewelSet = new HashSet<>();

        for (char jewel : jewels.toCharArray()) {
            jewelSet.add(jewel);
        }

        // Count how many stones are jewels
        int count = 0;
        for (char stone : stones.toCharArray()) {
            if (jewelSet.contains(stone)) {
                count++;
            }
        }

        return count;
    }

    public static int RingsAndRods(String rings) {
        Set<Character>[] rods = new Set[10];

        // Initialize the sets for each rod
        for (int i = 0; i < 10; i++) {
            rods[i] = new HashSet<>();
        }

        // Iterate through the rings string
        for (int i = 0; i < rings.length(); i += 2) {
            char color = rings.charAt(i);     // Color of the ring (R, G, B)
            int rod = rings.charAt(i + 1) - '0'; // Rod number (convert char to int)

            // Add the color to the corresponding rod's set
            rods[rod].add(color);
        }

        // Count the number of rods that have all three colors
        int count = 0;
        for (Set<Character> rod : rods) {
            if (rod.size() == 3) {
                count++;
            }
        }

        return count;
    }

    public static boolean CheckIfTheSentenceIsPangram(String sentence) {
        boolean[] seen = new boolean[26];

        // Iterate over each character in the sentence
        for (char c : sentence.toCharArray()) {
            // Mark the letter as seen
            seen[c - 'a'] = true;
        }

        // Check if all letters are marked as seen
        for (boolean letter : seen) {
            if (!letter) {
                return false;  // If any letter is not seen, it's not a pangram
            }
        }

        return true;
    }

    public static int UniqueMorseCodeWords(String[] words) {
        String[] morseCode = {".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."};

        // HashSet to track unique Morse code transformations
        HashSet<String> uniqueTransformations = new HashSet<>();

        // Iterate over each word in the input array
        for (String word : words) {
            StringBuilder morseWord = new StringBuilder();

            // Convert each character of the word to Morse code and append to morseWord
            for (char c : word.toCharArray()) {
                int index = c - 'a'; // Calculate the index of the letter in the alphabet
                morseWord.append(morseCode[index]);
            }

            // Add the Morse code transformation to the HashSet
            uniqueTransformations.add(morseWord.toString());
        }

        // Return the number of unique Morse code transformations
        return uniqueTransformations.size();
    }

    public static String DestinationCity(List<List<String>> paths) {
        // Set to track cities that have outgoing paths
        Set<String> citiesWithOutgoingPaths = new HashSet<>();

        // Loop through each path and add the city with outgoing path (cityAi) to the set
        for (List<String> path : paths) {
            citiesWithOutgoingPaths.add(path.get(0)); // Add the cityAi (outgoing path)
        }

        // Now find the destination city (cityBi) that does not have any outgoing path
        for (List<String> path : paths) {
            String cityB = path.get(1); // cityBi
            if (!citiesWithOutgoingPaths.contains(cityB)) {
                return cityB; // This is the destination city
            }
        }

        return ""; // Return an empty string if no destination city is found (though the problem guarantees one exists)
    }
    //endregion

    //region Sorting
    public static int[] HowManyNumbersAreSmallerThanTheCurrentNumber1(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];

        // Create a copy of the array and sort it
        int[] sorted = nums.clone();
        Arrays.sort(sorted);

        // Use a hash map to store the count of smaller elements for each unique number
        Map<Integer, Integer> countMap = new HashMap<>();

        // Traverse the sorted array and record the count of smaller elements
        for (int i = 0; i < n; i++) {
            if (!countMap.containsKey(sorted[i])) {
                countMap.put(sorted[i], i);  // The index is the count of smaller elements
            }
        }

        // For each element in the original array, find how many elements are smaller
        for (int i = 0; i < n; i++) {
            result[i] = countMap.get(nums[i]);
        }

        return result;
    }

    public static int KeepMultiplyingFoundValuesByTwo(int[] nums, int original) {
        while (contains(nums, original)) {
            original *= 2;
        }
        return original;
    }

    public static boolean MakeTwoArraysEqualByReversingSubArrays(int[] target, int[] arr) {
        Arrays.sort(target);
        Arrays.sort(arr);
        return Arrays.equals(target, arr);
    }

    public static int[] IntersectionOfTwoArrays1(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();

        // Add elements from nums1 to set1
        for (int num : nums1) {
            set1.add(num);
        }

        // Add elements from nums2 to set2
        for (int num : nums2) {
            set2.add(num);
        }

        // Find the intersection of both sets
        set1.retainAll(set2);  // retain only elements that are in both sets

        // Convert the set to an array and return
        int[] result = new int[set1.size()];
        int index = 0;
        for (int num : set1) {
            result[index++] = num;
        }

        return result;
    }

    public static int[] SortArrayByIncreasingFrequency(int[] nums) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        List<Integer> numList = new ArrayList<>();
        for (int num : nums) {
            numList.add(num);
        }

        numList.sort((a, b) -> {
            int freqA = frequencyMap.get(a);
            int freqB = frequencyMap.get(b);
            if (freqA != freqB) {
                return Integer.compare(freqA, freqB);
            } else {
                return Integer.compare(b, a);
            }
        });

        for (int i = 0; i < nums.length; i++) {
            nums[i] = numList.get(i);
        }

        return nums;
    }
    //endregion

    //region Counting
    public static int NumberOfGoodPairs(int[] nums) {
        HashMap<Integer, Integer> freqMap = new HashMap<>();
        int count = 0;

        // Traverse the array and update the frequency map
        for (int num : nums) {
            // For each number, the number of good pairs is the current frequency of that number
            count += freqMap.getOrDefault(num, 0);
            freqMap.put(num, freqMap.getOrDefault(num, 0) + 1);
        }

        return count;
    }

    public static int[] HowManyNumbersAreSmallerThanTheCurrentNumber2(int[] nums){
        return HowManyNumbersAreSmallerThanTheCurrentNumber1(nums);
    }

    public static int CountNumberOfPairsWithAbsoluteDifferenceK(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        int res = 0;

        for (int num : nums) {
            if (map.containsKey(num - k)) {
                res += map.get(num - k);
            }
            if (map.containsKey(num + k)) {
                res += map.get(num + k);
            }
            map.put(num, map.getOrDefault(num, 0) + 1);
        }


        return res;
    }

    public static String IncreasingDecreasingString(String s) {
        StringBuilder ans = new StringBuilder();
        int[] count = new int[26];
        for (char c : s.toCharArray())
            ++count[c - 'a'];
        while (ans.length() < s.length()) {
            add(count, ans, true);
            add(count, ans, false);
        }
        return ans.toString();
    }

    public static boolean CheckIfAllCharactersHaveEqualNumberOfOccurrences(String s) {
        HashMap<Character, Integer> frequencyMap = new HashMap<>();

        // Step 1: Count the frequency of each character
        for (char c : s.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }

        // Step 2: Get the frequency of the first character
        int frequency = frequencyMap.values().iterator().next();

        // Step 3: Check if all characters have the same frequency
        for (int count : frequencyMap.values()) {
            if (count != frequency) {
                return false;
            }
        }

        return true;
    }
    //endregion

    //region Bit Manipulation
    public static int CountTheNumberOfConsistentStrings(String allowed, String[] words) {
        HashSet<Character> allowedSet = new HashSet<>();
        for (char c : allowed.toCharArray()) {
            allowedSet.add(c);
        }

        int count = 0;

        // Step 2: Iterate through each word and check if it's consistent
        for (String word : words) {
            boolean isConsistent = true;

            // Step 3: Check if all characters in the word are in the allowed set
            for (char c : word.toCharArray()) {
                if (!allowedSet.contains(c)) {
                    isConsistent = false;
                    break;
                }
            }

            // If the word is consistent, increment the count
            if (isConsistent) {
                count++;
            }
        }

        return count;
    }

    public static boolean DivideArrayIntoEqualPairs(int[] nums) {
        HashMap<Integer, Integer> countMap = new HashMap<>();
        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }

        // Step 2: Check if every element appears an even number of times
        for (int count : countMap.values()) {
            if (count % 2 != 0) {
                return false;
            }
        }

        // Step 3: If all counts are even, return true
        return true;
    }

    public static int[] SetMismatch(int[] nums) {
        int n = nums.length;
        long expectedSum = (long) n * (n + 1) / 2;
        long expectedSumOfSquares = (long) n * (n + 1) * (2L * n + 1) / 6;
        long actualSum = 0;
        long actualSumOfSquares = 0;

        for (int num : nums) {
            actualSum += num;
            actualSumOfSquares += (long) num * num;
        }

        long sumDifference = actualSum - expectedSum;
        long squareDifference = actualSumOfSquares - expectedSumOfSquares;
        long duplicatePlusMissing = squareDifference / sumDifference;
        int duplicate = (int) ((sumDifference + duplicatePlusMissing) / 2);
        int missing = (int) (duplicate - sumDifference);

        return new int[]{duplicate, missing};
    }

    public static int MissingNumber(int[] nums) {
        int n = nums.length;
        int expectedSum = n * (n + 1) / 2;
        int actualSum = 0;

        for (int num : nums) {
            actualSum += num;
        }

        return expectedSum - actualSum;
    }

    public int CountTripletsThatCanFormTwoArraysOfEqualXOR(int[] arr) {
        int count = 0;
        int n = arr.length;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int a = 0;
                for (int k = i; k < j; k++) {
                    a ^= arr[k];
                }

                int b = 0;
                for (int k = j; k < n; k++) {
                    b ^= arr[k];
                    if (a == b) {
                        count++;
                    }
                }
            }
        }

        return count;
    }
    //endregion

    //region Binary Search
    public static int[] IntersectionOfTwoArrays2(int[] nums1, int[] nums2) {
        return IntersectionOfTwoArrays1(nums1,nums2);
    }

    public static boolean CheckIfNAndItsDoubleExist1(int[] arr) {
        HashSet<Integer> seen = new HashSet<>();

        for (int num : arr) {
            // Check if we already have the element or its double or half
            if (seen.contains(num * 2) || (num % 2 == 0 && seen.contains(num / 2))) {
                return true;
            }
            // Add current number to the set for future comparisons
            seen.add(num);
        }

        return false;
    }

    public int[] IntersectionOfTwoArrays3(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums1) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        // Step 2: Find the intersection and store the result
        List<Integer> result = new ArrayList<>();
        for (int num : nums2) {
            if (map.containsKey(num) && map.get(num) > 0) {
                result.add(num);
                map.put(num, map.get(num) - 1); // Decrement the count
            }
        }

        // Convert result list to an array
        int[] intersection = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            intersection[i] = result.get(i);
        }

        return intersection;
    }

    public static int[] NumberOfFlowersInFullBloom(int[][] flowers, int[] persons) {
        Queue<int[]> pq = new PriorityQueue<>((a,b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

        for(int i = 0; i < persons.length; i++){
            pq.offer(new int[]{persons[i], 1, i});
        }

        for(int[] flower : flowers){
            pq.offer(new int[]{flower[0], 0});
            pq.offer(new int[]{flower[1], 2});
        }

        int[] ret = new int[persons.length];
        int numEvents = pq.size();
        int blooms = 0;

        for(int i = 0; i < numEvents; i++){
            int[] cur = pq.poll();

            if(cur[1] == 0){
                blooms++;
            }else if(cur[1] == 2){
                blooms--;
            }else{
                int personIndex = cur[2];
                ret[personIndex] = blooms;
            }
        }
        return ret;
    }
    //endregion

    //region Sliding Window
    public static int SubstringsOfSizeThreeWithDistinctCharacters(String s) {
        int count = 0;
        // Loop through the string and check substrings of length 3
        for (int i = 0; i < s.length() - 2; i++) {
            String substring = s.substring(i, i + 3);
            // Check if the substring has all unique characters
            if (substring.charAt(0) != substring.charAt(1) &&
                    substring.charAt(0) != substring.charAt(2) &&
                    substring.charAt(1) != substring.charAt(2)) {
                count++;
            }
        }
        return count;
    }

    public static boolean ContainsDuplicate2(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            // If the number is already in the map, check the index difference
            if (map.containsKey(nums[i]) && i - map.get(nums[i]) <= k) {
                return true;
            }
            // Update the map with the current index for the number
            map.put(nums[i], i);
        }

        // No such pair found
        return false;
    }

    public static int NumberOfSubstringsContainingAllThreeCharacters(String s) {
        int n = s.length();
        int count = 0;
        HashMap<Character, Integer> map = new HashMap<>();

        int left = 0;

        for (int right = 0; right < n; right++) {
            // Update the frequency of the current character
            map.put(s.charAt(right), map.getOrDefault(s.charAt(right), 0) + 1);

            // Check if the current window contains all three characters
            while (map.size() == 3) {
                // If so, add the number of valid substrings ending at 'right'
                count += n - right;

                // Shrink the window from the left
                char leftChar = s.charAt(left);
                map.put(leftChar, map.get(leftChar) - 1);
                if (map.get(leftChar) == 0) {
                    map.remove(leftChar);
                }
                left++;
            }
        }

        return count;
    }

    public static double[] SlidingWindowMedian(int[] nums, int k) {
        Comparator<Integer> comparator = (a, b) -> nums[a] != nums[b] ? Integer.compare(nums[a], nums[b]) : a - b;
        TreeSet<Integer> left = new TreeSet<>(comparator.reversed());
        TreeSet<Integer> right = new TreeSet<>(comparator);

        Supplier<Double> median = (k % 2 == 0) ?
                () -> ((double) nums[left.first()] + nums[right.first()]) / 2 :
                () -> (double) nums[right.first()];

        // balance lefts size and rights size (if not equal then right will be larger by one)
        Runnable balance = () -> { while (left.size() > right.size()) right.add(left.pollFirst()); };

        double[] result = new double[nums.length - k + 1];

        for (int i = 0; i < k; i++) left.add(i);
        balance.run(); result[0] = median.get();

        for (int i = k, r = 1; i < nums.length; i++, r++) {
            // remove tail of window from either left or right
            if(!left.remove(i - k)) right.remove(i - k);

            // add next num, this will always increase left size
            right.add(i); left.add(right.pollFirst());

            // rebalance left and right, then get median from them
            balance.run(); result[r] = median.get();
        }

        return result;
    }

    public static String MinimumWindowSubString(String s, String t) {
        if (s.length() < t.length()) return "";

        HashMap<Character, Integer> tCount = new HashMap<>();
        for (char c : t.toCharArray()) {
            tCount.put(c, tCount.getOrDefault(c, 0) + 1);
        }

        HashMap<Character, Integer> windowCount = new HashMap<>();

        int left = 0, right = 0;
        int required = tCount.size();
        int formed = 0;
        int minLength = Integer.MAX_VALUE;
        int[] result = {-1, -1};

        while (right < s.length()) {

            char c = s.charAt(right);
            windowCount.put(c, windowCount.getOrDefault(c, 0) + 1);

            if (tCount.containsKey(c) && windowCount.get(c).intValue() == tCount.get(c).intValue()) {
                formed++;
            }

            while (left <= right && formed == required) {
                c = s.charAt(left);

                if (right - left + 1 < minLength) {
                    minLength = right - left + 1;
                    result[0] = left;
                    result[1] = right;
                }

                windowCount.put(c, windowCount.get(c) - 1);
                if (tCount.containsKey(c) && windowCount.get(c) < tCount.get(c)) {
                    formed--;
                }

                left++;
            }

            right++;
        }

        return minLength == Integer.MAX_VALUE ? "" : s.substring(result[0], result[1] + 1);
    }
    //endregion

    //region Two Pointer
    public static int[] IntersectionOfTwoArrays4(int[] nums1, int[] nums2) {
        return IntersectionOfTwoArrays1(nums1, nums2);
    }

    public static boolean CheckIfNAndItsDoubleExist2(int[] arr){
        return CheckIfNAndItsDoubleExist1(arr);
    }

    public static ListNode IntersectionOfTwoLinkedLists(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }

        // Step 1: Get the lengths of both linked lists
        ListNode currA = headA;
        ListNode currB = headB;
        int lengthA = 0, lengthB = 0;

        // Calculate length of list A
        while (currA != null) {
            lengthA++;
            currA = currA.next;
        }

        // Calculate length of list B
        while (currB != null) {
            lengthB++;
            currB = currB.next;
        }

        // Step 2: Align the starting points of both lists
        currA = headA;
        currB = headB;

        // Move the pointer of the longer list ahead by the difference in lengths
        if (lengthA > lengthB) {
            for (int i = 0; i < lengthA - lengthB; i++) {
                currA = currA.next;
            }
        } else if (lengthB > lengthA) {
            for (int i = 0; i < lengthB - lengthA; i++) {
                currB = currB.next;
            }
        }

        // Step 3: Move both pointers together and check for intersection
        while (currA != null && currB != null) {
            if (currA == currB) {
                return currA; // They intersect here
            }
            currA = currA.next;
            currB = currB.next;
        }

        // No intersection found
        return null;
    }

    public static boolean IsNumberHappy(int n) {
        Set<Integer> seen = new HashSet<>();

        // Loop until we either find 1 or detect a cycle
        while (n != 1) {
            if (seen.contains(n)) {
                return false; // We encountered a cycle
            }
            seen.add(n);

            // Calculate sum of squares of digits of n
            n = sumOfSquares(n);
        }

        // If we reached 1, it's a happy number
        return true;
    }

    public static int[] IntersectionOfTwoArrays5(int[] nums1, int[] nums2) {
        return IntersectionOfTwoArrays1(nums1, nums2);
    }
    //endregion

    //region Divide and Conquer
    public static int[] TopKFrequentElements(int[] nums, int k) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        // Step 2: Use a priority queue (min-heap) to keep the top k frequent elements
        PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>(
                (a, b) -> a.getValue() - b.getValue() // compare based on frequency
        );

        // Step 3: Add entries to the heap
        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            minHeap.offer(entry);
            if (minHeap.size() > k) {
                minHeap.poll(); // Remove the element with the smallest frequency
            }
        }

        // Step 4: Extract the k most frequent elements from the heap
        int[] result = new int[k];
        int i = 0;
        while (!minHeap.isEmpty()) {
            result[i++] = minHeap.poll().getKey();
        }

        return result;
    }

    public static TreeNode ConstructBinaryTreeFromPreorderAndInorderTraversal(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, inorder, 0, 0, inorder.length - 1);
    }
    //endregion

    //region DP Based
    public static int CountSubstringsThatDifferByOneCharacter(String s, String t) {
        int count = 0;
        int m = s.length();
        int n = t.length();

        // Iterate over all possible starting points for substrings in s
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j <= m; j++) {
                String subS = s.substring(i, j); // Substring of s
                for (int k = 0; k < n - (j - i) + 1; k++) {
                    String subT = t.substring(k, k + (j - i)); // Substring of t
                    if (canTransform(subS, subT)) {
                        count++;
                    }
                }
            }
        }

        return count;
    }

    public static int LongestStringChain(String[] words) {
        Arrays.sort(words, (a, b) -> a.length() - b.length());

        // HashMap to store the longest chain ending with each word
        Map<String, Integer> dp = new HashMap<>();
        int longestChain = 1;

        // Iterate through each word
        for (String word : words) {
            dp.put(word, 1);  // The word itself is a valid chain of length 1
            // Try removing each character from the word
            for (int i = 0; i < word.length(); i++) {
                String predecessor = word.substring(0, i) + word.substring(i + 1);
                // If the predecessor exists in dp, update the chain length
                if (dp.containsKey(predecessor)) {
                    dp.put(word, Math.max(dp.get(word), dp.get(predecessor) + 1));
                }
            }
            // Update the longest chain found so far
            longestChain = Math.max(longestChain, dp.get(word));
        }

        return longestChain;
    }

    public static long TotalAppealOfAString(String s) {
        long res = 0;
        long cur = 0;
        long[] prev = new long[26];
        for (int i = 0; i < s.length(); ++i) {
            final var l = prev[s.charAt(i) - 'a'];
            cur += i + 1 - l;
            prev[s.charAt(i) - 'a'] = i + 1;
            res += cur;
        }
        return res;
    }
    //endregion

    //region Tree Based
    public static void FindElementsInAContaminatedBinaryTree() {
        TreeNode root = new TreeNode(-1);
        root.left = new TreeNode(-1);
        root.right = new TreeNode(-1);
        root.left.left = new TreeNode(-1);
        root.left.right = new TreeNode(-1);
        root.right.left = new TreeNode(-1);
        root.right.right = new TreeNode(-1);

        // Initialize the FindElements class with the contaminated tree
        FindElements obj = new FindElements(root);

        // Example queries
        System.out.println(obj.find(1)); // Output: true
        System.out.println(obj.find(2)); // Output: true
        System.out.println(obj.find(3)); // Output: true
        System.out.println(obj.find(4)); // Output: false
    }

    public static TreeNode CreateBinaryTreeFromDescriptions(int[][] descriptions) {
        HashMap<Integer, TreeNode> map = new HashMap<>();
        HashSet<Integer> children = new HashSet<>();

        for (int[] description : descriptions) {
            int parentVal = description[0];
            int childVal = description[1];
            boolean isLeft = description[2] == 1;

            map.putIfAbsent(parentVal, new TreeNode(parentVal));
            map.putIfAbsent(childVal, new TreeNode(childVal));

            TreeNode parent = map.get(parentVal);
            TreeNode child = map.get(childVal);

            if (isLeft) {
                parent.left = child;
            } else {
                parent.right = child;
            }

            children.add(childVal);
        }

        for (int parentVal : map.keySet()) {
            if (!children.contains(parentVal)) {
                return map.get(parentVal);
            }
        }

        return null;
    }

    public static TreeNode LowestCommonAncestorOfDeepestLeaves(TreeNode root) {
        dfs(root, 0);  // Start DFS from root with depth 0
        return lcaNode;
    }

    public static TreeNode SmallestSubtreeWithAllTheDeepestNodes(TreeNode root) {
        dfs(root, 0);  // Start DFS from root with depth 0
        return result;
    }

    public static void ThroneInheritance() {
        ThroneInheritance throneInheritance = new ThroneInheritance("king");

        throneInheritance.birth("king", "Alice");
        throneInheritance.birth("king", "Bob");
        throneInheritance.birth("Alice", "Jack");

        System.out.println(throneInheritance.getInheritanceOrder()); // ["king", "Alice", "Jack", "Bob"]

        throneInheritance.death("Alice");
        System.out.println(throneInheritance.getInheritanceOrder()); // ["king", "Jack", "Bob"]

        throneInheritance.birth("Bob", "Lily");
        System.out.println(throneInheritance.getInheritanceOrder()); // ["king", "Jack", "Bob", "Lily"]
    }
    //endregion

    //region Matrix
    public static int FlipColumnsForMaximumNumberOfEqualRows(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        // HashMap to store the frequency of row patterns
        Map<String, Integer> rowPatterns = new HashMap<>();

        // Iterate over each row
        for (int[] intigers : matrix) {
            // Generate the original and flipped versions of the row
            StringBuilder originalRow = new StringBuilder();
            StringBuilder flippedRow = new StringBuilder();

            for (int j = 0; j < n; j++) {
                originalRow.append(intigers[j]);
                flippedRow.append(intigers[j] == 0 ? 1 : 0); // Flip the bit
            }

            // Add both original and flipped versions to the map (key = row pattern, value = frequency)
            String original = originalRow.toString();
            String flipped = flippedRow.toString();

            rowPatterns.put(original, rowPatterns.getOrDefault(original, 0) + 1);
            rowPatterns.put(flipped, rowPatterns.getOrDefault(flipped, 0) + 1);
        }

        // Find the maximum frequency
        int maxRows = 0;
        for (int count : rowPatterns.values()) {
            maxRows = Math.max(maxRows, count);
        }

        return maxRows;
    }

    public static boolean CheckIfEveryRowAndColumnContainsAllNumbers(int[][] matrix) {
        int n = matrix.length;

        // Check rows
        for (int[] ints : matrix) {
            boolean[] seen = new boolean[n];
            for (int j = 0; j < n; j++) {
                int num = ints[j];
                if (num < 1 || num > n || seen[num - 1]) {
                    return false; // invalid number or duplicate in row
                }
                seen[num - 1] = true;
            }
        }

        // Check columns
        for (int j = 0; j < n; j++) {
            boolean[] seen = new boolean[n];
            for (int[] ints : matrix) {
                int num = ints[j];
                if (num < 1 || num > n || seen[num - 1]) {
                    return false; // invalid number or duplicate in column
                }
                seen[num - 1] = true;
            }
        }

        return true;
    }

    public static int NumberOfSubMatricesThatSumToTarget(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int result = 0;

        // Iterate over all pairs of rows
        for (int row1 = 0; row1 < m; row1++) {
            int[] colSums = new int[n];  // This will store the sum of columns between rows row1 and row2

            for (int row2 = row1; row2 < m; row2++) {
                // Update column sums for the current row pair (row1, row2)
                for (int col = 0; col < n; col++) {
                    colSums[col] += matrix[row2][col];
                }

                // Now we need to find the number of subarrays with sum equal to target in colSums
                HashMap<Integer, Integer> sumCount = new HashMap<>();
                sumCount.put(0, 1);  // To handle the case where a subarray itself equals target

                int currentSum = 0;

                for (int sum : colSums) {
                    currentSum += sum;
                    if (sumCount.containsKey(currentSum - target)) {
                        result += sumCount.get(currentSum - target);
                    }
                    sumCount.put(currentSum, sumCount.getOrDefault(currentSum, 0) + 1);
                }
            }
        }

        return result;
    }
    //endregion

    //region Private Methods
    private static boolean contains(int[] nums, int value) {
        for (int num : nums) {
            if (num == value) {
                return true;
            }
        }
        return false;
    }

    private static void add(int[] cnt, StringBuilder ans, boolean asc) {
        for (int i = 0; i < 26; ++i) {
            int j = asc ? i : 25 - i;
            if (cnt[j]-- > 0)
                ans.append((char)(j + 'a'));
        }
    }

    private static int sumOfSquares(int n) {
        int sum = 0;
        while (n > 0) {
            int digit = n % 10; // Get the last digit
            sum += digit * digit; // Add the square of the digit
            n /= 10; // Remove the last digit
        }
        return sum;
    }

    private static TreeNode buildTreeHelper(int[] preorder, int[] inorder, int preStart, int inStart, int inEnd) {
        // Base case: if there are no elements to construct the tree
        if (preStart > preorder.length - 1 || inStart > inEnd) {
            return null;
        }

        // The first element in preorder is the root
        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);

        // Find the index of the root in inorder
        int rootIndex = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == rootVal) {
                rootIndex = i;
                break;
            }
        }

        // Recursively build the left and right subtrees
        root.left = buildTreeHelper(preorder, inorder, preStart + 1, inStart, rootIndex - 1);
        root.right = buildTreeHelper(preorder, inorder, preStart + (rootIndex - inStart + 1), rootIndex + 1, inEnd);

        return root;
    }

    private static boolean canTransform(String subS, String subT) {
        int diffCount = 0;
        for (int i = 0; i < subS.length(); i++) {
            if (subS.charAt(i) != subT.charAt(i)) {
                diffCount++;
            }
            if (diffCount > 1) {
                return false;
            }
        }
        return diffCount == 1;
    }

    private static int dfs(TreeNode node, int depth) {
        if (node == null) return depth;

        // Recurse to left and right
        int leftDepth = dfs(node.left, depth + 1);
        int rightDepth = dfs(node.right, depth + 1);

        // Update max depth if we find a deeper level
        maxDepth = Math.max(maxDepth, Math.max(leftDepth, rightDepth));

        // If both sides are at the maximum depth, this node is a candidate for LCA
        if (leftDepth == maxDepth && rightDepth == maxDepth) {
            lcaNode = node;
        }

        // Return the current node's depth
        return Math.max(leftDepth, rightDepth);
    }
    //endregion
}
