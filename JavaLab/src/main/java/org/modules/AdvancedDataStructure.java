package org.modules;

import org.dto.*;
import org.helpers.MaxFlow;

import java.util.*;
import java.util.stream.Collectors;

public class AdvancedDataStructure {

    //region Constants And Variables
    private static final int NOT_COVERED = 0;
    private static final int COVERED_NO_WATCHMAN = 1;
    private static final int HAS_WATCHMAN = 2;
    private static int watchmenCount = 0;
    private static final int MOD = 1000000007;
    private static long[] fact;
    private static long[] invFact;
    private static int minDifference = Integer.MAX_VALUE;
    private static Integer prevValue = null;
    private static final List<Integer> values = new ArrayList<>();
    private static final List<Integer> medianData = new ArrayList<>();
    private static final PriorityQueue<Integer> lowerHalfHeap = new PriorityQueue<>(Collections.reverseOrder());
    private static final PriorityQueue<Integer> upperHalfHeap = new PriorityQueue<>();
    private static Pair<Integer, Integer> temp;
    //endregion

    //region Linked Lists 1
    public static ListNode SumOfLinkedLists(ListNode l1, ListNode l2) {
        // Reverse both lists to make addition easier (LSB first)
        l1 = ReverseList(l1);
        l2 = ReverseList(l2);

        // Create a new linked list for the result
        ListNode dummyHead = new ListNode(0);
        ListNode current = dummyHead;
        int carry = 0;

        // Add the numbers as long as there are nodes in either list or there's a carry
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry;

            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }

            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }

            carry = sum / 10;  // Calculate the carry
            current.next = new ListNode(sum % 10);  // Create a new node for the digit
            current = current.next;
        }

        // Reverse the result list to restore the correct order
        return ReverseList(dummyHead.next);

    }

    public static Node DesignNode(int initialValue) {
        return new Node(initialValue);
    }

    public static void PrintMiddleNode(Node head) {
        // Initialize two pointers: slow and fast
        Node slow = head;
        Node fast = head;

        // Traverse the list with fast moving twice as fast as slow
        while (fast != null && fast.next != null && fast.next.next != null) {
            slow = slow.next;        // Move slow by one
            fast = fast.next.next;   // Move fast by two
        }

        // At this point, slow is at the middle node
        System.out.println(slow != null ? slow.data : 0);
    }

    public static ListNode ReverseList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    //endregion

    //region Linked Lists 2
    public static boolean PalindromicLinkedList(ListNode head) {
        if (head == null || head.next == null) {
            return true; // A single node or empty list is a palindrome
        }

        // Step 1: Find the middle of the linked list
        ListNode slow = head;
        ListNode fast = head;
        Stack<Integer> stack = new Stack<>();

        // Step 2: Push the first half onto the stack
        while (fast != null && fast.next != null) {
            stack.push(slow.val);
            slow = slow.next;
            fast = fast.next.next;
        }

        // If the list has an odd number of elements, skip the middle element
        if (fast != null) {
            slow = slow.next;
        }

        // Step 3: Compare the second half with the stack
        while (slow != null) {
            if (stack.pop() != slow.val) {
                return false; // Not a palindrome
            }
            slow = slow.next;
        }

        return true; // All values matched, it is a palindrome
    }

    public static ListNode AddNodeFirstMiddleAndLast(ListNode head, int val) {
        if (head == null) return null;

        //1. Insert at front, front node will have 'prev' value as null, it is head itself
        ListNode newNodeBeginning = new ListNode(val);
        newNodeBeginning.next = head;
        head.prev = newNodeBeginning;
        head = newNodeBeginning;

        //2. Find last, in-between any -ve value node, then filter it-out
        ListNode curr = head;
        ListNode last = null;
        while(curr != null){
            if(curr.val < 0) {
                curr.prev.next = curr.next;
            }

            if(curr.next == null) {
                if(curr.val > 0) {
                    last = curr;
                } else {
                    last = curr.prev;
                }
            }

            curr = curr.next;
        }

        ListNode newEndNode = new ListNode(val);
        last.next = newEndNode;
        newEndNode.prev = last;

        // Find the length of updated node
        int length = getLength(head);

        // Create node for middle
        ListNode newNodeMiddle = new ListNode(val);

        // 3. Insert at the middle (ignoring negative values)
        int mid = length / 2;  // This gives the position just after the midpoint
        int count = 0;

        // Move to the midpoint, skipping negative nodes
        curr = head;
        while (curr != null && count < mid) {
            if (curr.val >= 0) {
                count++;
            }
            curr = curr.next;
        }

        // Insert at the middle position
        newNodeMiddle.prev = curr == null ? null : curr.prev;
        newNodeMiddle.next = curr;
        if (curr != null && curr.prev != null) {
            curr.prev.next = newNodeMiddle;
            curr.prev = newNodeMiddle;
        }

        return head;
    }

    public static ListNode RotateNodesByKNode(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;  // No rotation needed for empty or single-node lists
        }

        // Step 1: Get the length of the list
        int length = getLength(head);

        // Step 2: Find the actual number of rotations required (K % N)
        k = k % length;
        if (k == 0) {
            return head;  // No rotation needed if k is 0
        }

        // Step 3: Move to the (N - K)th node, which will be the new tail after rotation
        ListNode newTail = head;
        for (int i = 1; i < length - k; i++) {
            newTail = newTail.next;
        }

        // Step 4: Set new head and rearrange pointers
        ListNode newHead = newTail.next;
        newTail.next = null;  // New tail points to null
        newHead.prev = null;  // New head's prev should be null

        // Step 5: Traverse to the last node and connect it to the old head
        ListNode last = newHead;
        while (last.next != null) {
            last = last.next;
        }
        last.next = head;
        head.prev = last;

        return newHead;
    }

    public static ListNode MergeTwoSortedLinkedLists(ListNode first, ListNode list2) {
        // Create a dummy node to act as the starting point of the merged list
        ListNode dummy = new ListNode(-1);
        ListNode current = dummy;

        // Traverse both lists and merge them
        while (first != null && list2 != null) {
            if (first.val <= list2.val) {
                current.next = first;
                first = first.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }

        // If there are remaining nodes in list1, append them
        if (first != null) {
            current.next = first;
        }

        // If there are remaining nodes in list2, append them
        if (list2 != null) {
            current.next = list2;
        }

        // Return the head of the merged list (next of dummy)
        return dummy.next;
    }

    public static ListNode MergeMultipleSortedLinkedList(List<ListNode> lists) {
        if(lists == null || lists.isEmpty()) {
            return null;
        }
        ListNode mergedList = lists.get(0);
        for(int i = 1; i < lists.size(); i++) {
            mergedList = MergeTwoSortedLinkedLists(mergedList, lists.get(i));
        }

        return mergedList;
    }
    //endregion

    //region Binary Trees
    public static int ZigzaggingThroughTheBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(zigzagDFS(root, true, 0), zigzagDFS(root, false, 0));
    }

    public static int TreeEscape(Node root) {
        if (root == null) return 0;

        int ramDepth = findLeftDepth(root);
        int shyamDepth = findRightDepth(root);

        return Integer.compare(ramDepth, shyamDepth);
    }

    public static int MaximumSumOfASubArrayInTopViewOfBinaryTree(TreeNode root){
        List<Integer> topView = getTopView(root);
        return maxSubArraySum(topView);
    }

    public static void Concatenate(Node root) {
        String result = largestNumberFromTree(root);
        System.out.println(result);
    }

    public static int WatchManOfBinaryTree(Node root) {
        if (dfs(root) == NOT_COVERED) {
            watchmenCount++;
        }
        return watchmenCount;
    }
    //endregion

    //region Binary Trees 1
    public static int IndexOfTarget(TreeNode root, int target) {
        int[] position = {0};  // To track the current in-order position
        return inOrderTraversal(root, target, position);
    }

    public static int findPeakElement(List<Integer> integerList) {
        int[] arr = integerList.stream().mapToInt(i -> i).toArray();
        int greatestPeak = Integer.MIN_VALUE;
        int peakIndex = -1;
        int n = arr.length;

        // Traverse the array to find the greatest peak element
        for (int i = 0; i < n; i++) {
            // Check if the current element is a peak
            if ((i == 0 || arr[i] >= arr[i - 1]) && (i == n - 1 || arr[i] >= arr[i + 1])) {
                // If this peak is greater than the current greatest peak, update it
                if (arr[i] > greatestPeak) {
                    greatestPeak = arr[i];
                    peakIndex = i;
                }
            }
        }
        return peakIndex;
    }

    public static int SweepersProblem(List<Integer> ar) {
        int[] arr = ar.stream().mapToInt(i -> i).toArray();
        int n = arr.length;
        precomputeFactorials(n);
        return countWays(arr);
    }

    public static int MinimumRiskPath(Node root) {
        inOrderTraversal(root);
        return minDifference;
    }

    public static boolean ThreeNumbersProduct(TreeNode root, int k) {
        inOrderTraversal(root);

        // Check if there are three elements such that their product equals k
        for (int i = 0; i < values.size(); i++) {
            for (int j = i + 1; j < values.size(); j++) {
                for (int l = j + 1; l < values.size(); l++) {
                    // Calculate the product of the three selected numbers
                    long product = (long) values.get(i) * values.get(j) * values.get(l);
                    // Check if the product equals k
                    if (product == k) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    //endregion

    //region Binary Trees 2
    public static int BookAllocationProblem(int[] books, int n, int students) {
        if (students > n) {
            return -1; // Not enough books for each student to have at least one
        }

        int low = getMax(books);
        int high = getSum(books);
        int result = high;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (isPossible(books, n, students, mid)) {
                result = mid; // Try for a smaller maximum
                high = mid - 1;
            } else {
                low = mid + 1; // Try for a larger maximum
            }
        }

        return result;
    }

    public static int IndexOfMaxElementInAnArray(int[] arr, int n) {
        if (n == 1) return 0;  // Single element case, it’s the peak by default

        int left = 0;
        int right = n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            // Check if mid is a peak
            boolean isLeftSmaller = (mid == 0 || arr[mid] > arr[mid - 1]);
            boolean isRightSmaller = (mid == n - 1 || arr[mid] > arr[mid + 1]);

            if (isLeftSmaller && isRightSmaller) {
                return mid;  // mid is the peak element
            } else if (mid > 0 && arr[mid] < arr[mid - 1]) {
                right = mid - 1;  // Move left
            } else {
                left = mid + 1;   // Move right
            }
        }

        return -1;  // No peak found that meets the criteria
    }

    public static int AggressiveCows(int n, int k, int[] stalls) {
        Arrays.sort(stalls);  // Ensure stalls are sorted

        int low = 1;
        int high = stalls[n - 1] - stalls[0];
        int result = 0;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (canPlaceCows(stalls, n, k, mid)) {
                result = mid; // feasible with `mid`, try for a larger distance
                low = mid + 1;
            } else {
                high = mid - 1; // not feasible with `mid`, try for a smaller distance
            }
        }

        return result;
    }

    public static ArrayList<Integer> SortTheNodesInBST(Node root) {
        ArrayList<Integer> unSortedList = new ArrayList<>();
        ArrayList<Integer> sortedList = new ArrayList<>();
        inOrderTraversal(root, unSortedList);
        Node outputRoot = null;
        for(int value : unSortedList) {
            outputRoot = insertBST(outputRoot,value);
        }
        inOrderTraversal(outputRoot, sortedList);
        return sortedList;
    }

    public static int FindingTheSizeOfTheLargest(Node root) {
        int[] maxBSTSize = new int[1];
        largestBSTHelper(root, maxBSTSize);
        return maxBSTSize[0];
    }
    //endregion

    //region Tries
    public static void InsertWordInTrie(String[] words, TrieNode root){
        for(String word : words) {
            insertIntoTrie(word,root);
        }
    }

    public static boolean SearchInTrie(String word, TrieNode root) {
        TrieNode currentNode = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int index = ch - 'a';

            if (currentNode.children[index] == null) {
                return false;
            }
            currentNode = currentNode.children[index];
        }
        return currentNode.isend;
    }

    public static List<String> AutoCompleteUsingTrie(String prefix, TrieNode root) {
        TrieNode currentNode = root;
        for (char ch : prefix.toCharArray()) {
            int index = ch - 'a';
            if (currentNode.children[index] == null) {
                return new ArrayList<>();  // No words with this prefix
            }
            currentNode = currentNode.children[index];
        }

        List<String> result = new ArrayList<>();
        getWordsFromNode(currentNode, new StringBuilder(prefix), result);

        Collections.sort(result);
        return result;
    }

    public static int XorPairInTrie(TrieNode root, int[] ar, int n){
        int maxXOR = 0;

        // Insert all numbers into the Trie
        for (int num : ar) {
            insertIntoTrie(num,root);
        }

        // For each number, find the max XOR with other numbers
        for (int num : ar) {
            maxXOR = Math.max(maxXOR, findMaxXOR(num,root));
        }

        return maxXOR;
    }

    public static void RemoveWord(String word, TrieNode root) {
        removeFromTrieNodeHelper(root,word,0);
    }
    //endregion

    //region Heaps 1
    public static int[] KthSmallestStream(int n, int k, int[] arr) {
        // Max-heap to store the k smallest elements
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        int[] result = new int[n];

        for (int i = 0; i < n; i++) {
            // Add the current element to the max-heap
            maxHeap.offer(arr[i]);

            // If heap size exceeds k, remove the largest element
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }

            // If heap size is less than k, kth smallest element doesn't exist
            if (maxHeap.size() < k) {
                result[i] = -1;
            } else {
                // The kth smallest element is the root of the max-heap
                result[i] = maxHeap.peek();
            }
        }

        return result;
    }

    public static int FactorsOfKthLargest(int n, int k) {
        List<Integer> factors = new ArrayList<>();

        // Find all factors of n
        for (int i = 1; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                factors.add(i);  // "i" is a factor
                if (i != n / i) {
                    factors.add(n / i);  // "n/i" is also a factor
                }
            }
        }

        // Sort factors in ascending order
        Collections.sort(factors);

        // Check if there are enough factors for the kth largest
        if (k > factors.size()) {
            return -1;  // Not enough factors
        }

        // Return the kth largest factor
        return factors.get(factors.size() - k);
    }

    public static String JumbleForTreasure(int[] digits, int n) {
        Arrays.sort(digits);

        // Create two numbers by alternately picking digits from sorted array
        StringBuilder num1 = new StringBuilder();
        StringBuilder num2 = new StringBuilder();

        for (int i = 0; i < digits.length; i++) {
            if (i % 2 == 0) {
                num1.append(digits[i]);
            } else {
                num2.append(digits[i]);
            }
        }

        // Convert the two number strings to integers and calculate their sum
        int number1 = Integer.parseInt(num1.toString());
        int number2 = Integer.parseInt(num2.toString());

        return String.valueOf(number1 + number2);

    }

    public static String KthLargestElement(String[] arr, int k) {
        // Min-heap to store the k largest elements
        PriorityQueue<String> minHeap = new PriorityQueue<>(new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                // Compare based on length first
                if (a.length() != b.length()) {
                    return Integer.compare(a.length(), b.length());
                }
                // If lengths are the same, compare lexicographically
                return a.compareTo(b);
            }
        });

        // Add elements to the heap
        for (String num : arr) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();  // Remove the smallest to keep only the k largest elements
            }
        }

        // The root of the heap is the kth largest element
        return minHeap.peek();
    }

    public static int[] LowestValueChallenge(int[] arr) {
        // Step 1: Create a min-heap (PriorityQueue in Java)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        // Step 2: Add all elements to the min-heap
        for (int num : arr) {
            minHeap.add(num);
        }

        int n = arr.length;
        int[] ans = new int[n];

        // Step 3: Simulate the game
        for (int i = 0; i < n / 2; i++) {
            // Alice removes the minimum element
            int alice = minHeap.poll();
            // Bob removes the next minimum element
            int bob = minHeap.poll();

            // Bob appends his element to ans first
            ans[2 * i] = bob;
            // Alice appends her element to ans next
            ans[2 * i + 1] = alice;
        }

        return ans;
    }
    //endregion

    //region Heaps 2
    public static int LightWar(Vector<Integer> stones) {
        // Step 1: Create a min-heap using a PriorityQueue
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        // Step 2: Add all stones to the min-heap
        minHeap.addAll(stones);

        // Step 3: Process the stones until one or none remain
        while (minHeap.size() > 1) {
            // Get the two lightest stones
            int stone1 = minHeap.poll(); // The lightest stone
            int stone2 = minHeap.poll(); // The second-lightest stone

            // If they are not equal, insert the remaining weight back into the heap
            if (stone1 != stone2) {
                minHeap.add(stone2 - stone1);
            }
        }

        // If there's one stone left, return its weight; otherwise return 0
        return minHeap.isEmpty() ? 0 : minHeap.poll();
    }

    public static List<Integer> RowWar(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;

        // List to hold the count of soldiers and corresponding row index
        List<int[]> soldierCount = new ArrayList<>();

        // Step 1: Count soldiers in each row
        for (int i = 0; i < m; i++) {
            int count = 0;
            for (int j = 0; j < n; j++) {
                count += mat[i][j];
            }
            soldierCount.add(new int[]{count, i}); // Store count and index
        }

        // Step 2: Sort the list based on soldiers count and index
        soldierCount.sort((a, b) -> {
            if (a[0] == b[0]) {
                return Integer.compare(a[1], b[1]); // Sort by index if counts are equal
            }
            return Integer.compare(a[0], b[0]); // Sort by soldier count
        });

        // Step 3: Collect the indices of the k weakest rows
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = soldierCount.get(i)[1]; // Get the index of the weak row
        }

        return Arrays.stream(result).boxed().collect(Collectors.toList());
    }

    public static int KthSmallestElementInASortedMatrix(int[][] matrix, int k){
        int n = matrix.length;
        int low = matrix[0][0];
        int high = matrix[n - 1][n - 1];

        while (low < high) {
            int mid = low + (high - low) / 2;
            int count = countLessEqual(matrix, mid);

            if (count < k) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        return low;
    }

    public static List<List<String>> GroupingSimilarItems(List<List<String>> items) {
        UnionFind uf = new UnionFind();
        Map<String, String> emailToName = new LinkedHashMap<>();

        // Step 1: Build union-find relationships
        for (List<String> item : items) {
            String name = item.get(0);
            String firstEmail = item.get(1);
            for (int i = 1; i < item.size(); i++) {
                String email = item.get(i);
                emailToName.put(email, name);  // Map each email to its name
                uf.union(firstEmail, email);  // Union all emails in the same item
            }
        }

        // Step 2: Collect emails by their root with LIFO order
        Map<String, Stack<String>> groupedEmails = new LinkedHashMap<>();
        for (String email : emailToName.keySet()) {
            String root = uf.find(email);
            groupedEmails.computeIfAbsent(root, k -> new Stack<>()).push(email);
        }

        // Step 3: Print the result in the desired format
        List<List<String>> result = new ArrayList<>();
        for (String root : groupedEmails.keySet()) {
            String name = emailToName.get(root);  // Get the name associated with the root email
            List<String> resultItem = new ArrayList<>();
            resultItem.add(name);
            while (!groupedEmails.get(root).isEmpty()) {
                String email = groupedEmails.get(root).pop();
                resultItem.add(email);
            }
            result.add(resultItem);
        }
        return result;
    }

    public static List<Long> MostFrequentId(int[] arr, int[] freq) {
        int n = arr.length;
        int[] ans = new int[n];
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

        for (int i = 0; i < n; i++) {
            int id = arr[i];
            int change = freq[i];

            // Get current count of ID, defaulting to 0 if it doesn't exist
            int currentCount = frequencyMap.getOrDefault(id, 0);
            int newCount = currentCount + change;

            // If the ID exists in the map, remove its old frequency from the heap
            if (currentCount > 0) {
                maxHeap.remove(currentCount);
            }

            // Update the frequency map with the new count
            if (newCount > 0) {
                frequencyMap.put(id, newCount);
                maxHeap.add(newCount);
            } else {
                frequencyMap.remove(id);
            }

            // Find the max frequency after the update
            ans[i] = maxHeap.isEmpty() ? 0 : maxHeap.peek();
        }

        List<Long> result = new ArrayList<>();
        for(int a : ans) {
            result.add((long) a);
        }
        return result;
    }
    //endregion

    //region Heaps 3
    public int MedianAtEveryStep(int currentNumber) {
        lowerHalfHeap.add(currentNumber);

        // Ensure the max-heap property (top of lowerHalfHeap <= top of upperHalfHeap)
        if (!upperHalfHeap.isEmpty() && lowerHalfHeap.peek() > upperHalfHeap.peek()) {
            upperHalfHeap.add(lowerHalfHeap.poll());
        }

        // Balance heaps if needed
        if (lowerHalfHeap.size() > upperHalfHeap.size() + 1) {
            upperHalfHeap.add(lowerHalfHeap.poll());
        } else if (upperHalfHeap.size() > lowerHalfHeap.size() + 1) {
            lowerHalfHeap.add(upperHalfHeap.poll());
        }

        // Return the median
        if (lowerHalfHeap.size() == upperHalfHeap.size()) {
            return (lowerHalfHeap.peek() + upperHalfHeap.peek()) / 2;
        } else if (lowerHalfHeap.size() > upperHalfHeap.size()) {
            return lowerHalfHeap.peek();
        } else {
            return upperHalfHeap.peek();
        }
    }

    public static void HeapImplementation(int[] arr, int n) {
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
    }

    public static int KLargestUsingHeap(int[] arr, int N, int K) {
        // Min-heap to store the k largest elements
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        // Iterate through the array
        for (int num : arr) {
            // Add the current number to the heap
            minHeap.offer(num);

            // If the heap size exceeds k, remove the smallest element
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        // Calculate the sum of elements in the heap
        int sum = 0;
        while (!minHeap.isEmpty()) {
            sum += minHeap.poll();
        }

        return sum;
    }

    public static int HackPrevention(String s) {
        int n = s.length();
        boolean hasLower = false, hasUpper = false, hasDigit = false;
        int replace = 0, oneSequence = 0, twoSequence = 0;

        // Check for character requirements
        for (char c : s.toCharArray()) {
            if (Character.isLowerCase(c)) hasLower = true;
            if (Character.isUpperCase(c)) hasUpper = true;
            if (Character.isDigit(c)) hasDigit = true;
        }

        // Count missing categories
        int missingTypes = (hasLower ? 0 : 1) + (hasUpper ? 0 : 1) + (hasDigit ? 0 : 1);

        // Handle repeating characters
        for (int i = 2; i < n; ) {
            if (s.charAt(i) == s.charAt(i - 1) && s.charAt(i) == s.charAt(i - 2)) {
                int length = 2;
                while (i < n && s.charAt(i) == s.charAt(i - 1)) {
                    length++;
                    i++;
                }

                replace += length / 3; // Number of replacements required
                if (length % 3 == 0) oneSequence++;
                else if (length % 3 == 1) twoSequence++;
            } else {
                i++;
            }
        }

        // Case 1: If the length is too short
        if (n < 6) {
            return Math.max(6 - n, missingTypes);
        }

        // Case 2: If the length is valid
        if (n <= 20) {
            return Math.max(replace, missingTypes);
        }

        // Case 3: If the length is too long
        int excessLength = n - 20;

        // Use deletions to optimize replacements
        replace -= Math.min(excessLength, oneSequence);
        replace -= Math.min(Math.max(0, excessLength - oneSequence), twoSequence * 2) / 2;
        replace -= Math.max(0, excessLength - oneSequence - 2 * twoSequence) / 3;

        return excessLength + Math.max(replace, missingTypes);

    }

    public static int StoneSmash(int[] stones) {
        // Create a max heap by using a priority queue with reversed order (max heap)
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

        // Add all stones to the max heap
        for (int stone : stones) {
            maxHeap.add(stone);
        }

        // Keep smashing stones until one or zero stones remain
        while (maxHeap.size() > 1) {
            // Extract the two largest stones
            int stone1 = maxHeap.poll();
            int stone2 = maxHeap.poll();

            // If the stones are not the same, push the difference back into the heap
            if (stone1 != stone2) {
                maxHeap.add(stone1 - stone2);
            }
        }

        // If there are no stones left, return 0, otherwise return the last stone's weight
        return maxHeap.isEmpty() ? 0 : maxHeap.poll();

    }
    //endregion

    //region Graphs 1
    public static double[] EvaluateDivision(List<List<String>> equations, List<Double> values, List<List<String>> queries) {
        UnionFind uf = new UnionFind();

        // Initialize union-find structure and process equations
        for (List<String> equation : equations) {
            String a = equation.get(0);
            String b = equation.get(1);
            uf.add(a);
            uf.add(b);
        }

        for (int i = 0; i < equations.size(); i++) {
            String a = equations.get(i).get(0);
            String b = equations.get(i).get(1);
            uf.union(a, b, values.get(i));
        }

        // Process queries and store results
        double[] result = new double[queries.size()];

        for (int i = 0; i < queries.size(); i++) {
            String c = queries.get(i).get(0);
            String d = queries.get(i).get(1);

            if (!uf.parent.containsKey(c) || !uf.parent.containsKey(d) || !uf.find(c).equals(uf.find(d))) {
                result[i] = -1.0;
            } else {
                result[i] = uf.weight.get(c) / uf.weight.get(d);
            }
        }

        return result;

    }

    public static int ShortestPathBetweenTwoNodes(int N, int[][] edges, int S, int D) {
        int E = edges.length;
        // Step 1: Build the graph as an adjacency list
        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.putIfAbsent(u, new ArrayList<>());
            graph.putIfAbsent(v, new ArrayList<>());
            graph.get(u).add(v);
            graph.get(v).add(u);  // Because the graph is undirected
        }

        // Step 2: BFS initialization
        Queue<Integer> queue = new LinkedList<>();
        int[] distance = new int[N];
        Arrays.fill(distance, -1);  // -1 means unvisited
        distance[S] = 0;  // Distance to source is 0

        queue.offer(S);

        // Step 3: BFS to find the shortest path
        while (!queue.isEmpty()) {
            int current = queue.poll();

            // Check all neighbors of the current node
            for (int neighbor : graph.get(current)) {
                if (distance[neighbor] == -1) {  // If neighbor hasn't been visited
                    distance[neighbor] = distance[current] + 1;
                    queue.offer(neighbor);

                    // If we reached the destination node, return the distance
                    if (neighbor == D) {
                        return distance[neighbor];
                    }
                }
            }
        }

        // If destination D is not reachable, return -1 (though the graph is connected)
        return -1;


    }

    public static List<String> DijkstraAlgorithm(int n, List<List<Pair<Integer, Integer>>> graph, int start) {
        Map<Integer, List<int[]>> graphMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graphMap.put(i, new ArrayList<>());
        }

        for (int i = 0; i < graph.size(); i++) {
            //System.out.println("Node " + i + " has neighbors:");
            for (Pair<Integer,Integer> neighbor : graph.get(i)) {
                //System.out.println("    " + neighbor.vertex + ", " + neighbor.weight);
                graphMap.get(i).add(new int[] {neighbor.vertex, neighbor.weight});
            }
        }

        int[] distances = dijkstraHelper(n, graphMap, start);
        List<String> result = new ArrayList<>();

        // Print results
        for (int i = 0; i < n; i++) {
            result.add(i + " " + (distances[i] == Integer.MAX_VALUE ? "Infinity" : distances[i]));
        }

        return result;
    }

    public static boolean DAG(int n, List<int[]> edges) {
        // Array to keep track of visited nodes: 0 = unvisited, 1 = visiting, 2 = visited
        int[] visited = new int[n];
        List<List<Integer>> graph = new ArrayList<>();
        for(int[] temp : edges) {
            List<Integer> tempList = new ArrayList<>();
            for(int val : temp) {
                tempList.add(val);
            }
            graph.add(tempList);
        }

        for (int i = 0; i < n; i++) {
            if (visited[i] == 0) {
                if (hasCycle(graph, i, visited)) {
                    return false;
                }
            }
        }
        return true;

    }

    public static int FindingStronglyConnectedComponents(int n, List<List<Integer>> a) {
        // Initialize graph and its transpose
        List<List<Integer>> graph = new ArrayList<>();
        List<List<Integer>> transposedGraph = new ArrayList<>();

        for (int i = 0; i <= n; i++) {
            graph.add(new ArrayList<>());
            transposedGraph.add(new ArrayList<>());
        }

        // Input edges
        for (List<Integer> temp : a) {
            int u = temp.get(0);
            int v = temp.get(1);
            graph.get(u).add(v);
            transposedGraph.get(v).add(u);
        }

        // Step 1: Perform DFS and store finish times in a stack
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[n + 1];

        for (int i = 1; i <= n; i++) {
            if (!visited[i]) {
                dfs(graph, i, visited, stack);
            }
        }

        // Step 2: Transpose the graph (already done during input)

        // Step 3: Perform DFS on the transposed graph
        Arrays.fill(visited, false);
        int sccCount = 0;

        while (!stack.isEmpty()) {
            int node = stack.pop();
            if (!visited[node]) {
                dfsTransposed(transposedGraph, node, visited);
                sccCount++; // Each DFS call corresponds to one SCC
            }
        }

        // Output the number of SCCs
        return sccCount;
    }
    //endregion

    //region Graphs 2
    public static int MaximumFlowInAFlowNetwork(int n, int m, int[][] edge) {

        MaxFlow.FlowNetwork network = new MaxFlow.FlowNetwork(n + 1);

        for (int[] temp : edge) {
            int u = temp[0];
            int v = temp[1];
            long w = temp[2];
            network.addEdge(u, v, w);
        }

        // Source node is 1, sink node is n
        int source = 1;

        MaxFlow.Dinic dinic = new MaxFlow.Dinic(network, source, n);
        return (int) dinic.maxFlow();
    }

    public static List<Integer> TopologicalSortingOfADirectedAcyclicGraph(int n, List<int[]> edges) {
        List<Integer> result = new ArrayList<>();
        int[] inDegree = new int[n]; // Array to store in-degrees of nodes
        List<List<Integer>> adjList = new ArrayList<>();

        // Initialize adjacency list
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }

        // Build adjacency list and in-degree array
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            adjList.get(u).add(v);
            inDegree[v]++;
        }

        // Add all nodes with in-degree 0 to the queue
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) {
                queue.add(i);
            }
        }

        // Process the nodes
        while (!queue.isEmpty()) {
            int node = queue.poll();
            result.add(node);

            // Reduce in-degree of adjacent nodes
            for (int neighbor : adjList.get(node)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.add(neighbor);
                }
            }
        }

        // Check if topological sorting is possible (no cycle)
        if (result.size() != n) {
            throw new IllegalArgumentException("The graph is not a DAG.");
        }

        return result;
    }

    public static int MinimumNumberOfEdgesToMakeADirectedGraphStronglyConnected(int[] source, int n, int m, int[] dest) {
        int[][] edges = new int[m][2];
        for (int i = 0; i < m; i++) {
            edges[i][0] = source[i];
        }
        for (int i = 0; i < m; i++) {
            edges[i][1] = dest[i];
        }

        return minEdgesToAdd(n, edges);
    }

    public static boolean CoursePlanner(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        // Populate the graph
        for (int[] prereq : prerequisites) {
            int course = prereq[0];
            int prerequisite = prereq[1];
            graph.get(course).add(prerequisite);
        }

        // Arrays to track the visit state of each course
        // 0 -> unvisited, 1 -> visiting, 2 -> visited
        int[] visitStatus = new int[numCourses];

        // Perform DFS for each course
        for (int i = 0; i < numCourses; i++) {
            if (visitStatus[i] == 0) {
                if (hasCycle(i, graph, visitStatus)) {
                    return false; // Cycle detected
                }
            }
        }

        return true;
    }

    public static List<List<Integer>> PacificAtlanticWaterFlow(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        List<List<Integer>> result = new ArrayList<>();

        // Boolean arrays to mark cells that can reach Pacific and Atlantic oceans
        boolean[][] pacificReach = new boolean[m][n];
        boolean[][] atlanticReach = new boolean[m][n];

        // Perform DFS from Pacific and Atlantic edges
        for (int i = 0; i < m; i++) {
            dfs(heights, pacificReach, i, 0); // Left edge (Pacific)
            dfs(heights, atlanticReach, i, n - 1); // Right edge (Atlantic)
        }

        for (int j = 0; j < n; j++) {
            dfs(heights, pacificReach, 0, j); // Top edge (Pacific)
            dfs(heights, atlanticReach, m - 1, j); // Bottom edge (Atlantic)
        }

        // Find cells that can reach both oceans
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacificReach[i][j] && atlanticReach[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }

        return result;
    }
    //endregion

    //region Graphs 3
    public static int OptimalFlightCost(int n, int[][] flights, int src, int dst, int k) {
        // Step 1: Build the graph (adjacency list representation)
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] flight : flights) {
            graph.putIfAbsent(flight[0], new ArrayList<>());
            graph.get(flight[0]).add(new int[]{flight[1], flight[2]});
        }

        // Step 2: Priority queue to store the current cost, city, and number of stops
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[]{0, src, 0});  // {cost, city, stops}

        // Step 3: Visited array to track the minimum cost at each city with a certain number of stops
        int[][] visited = new int[n][k + 2];  // k+2 to handle up to k stops + the first 0 stops
        for (int[] arr : visited) {
            Arrays.fill(arr, Integer.MAX_VALUE);
        }
        visited[src][0] = 0;

        // Step 4: Perform BFS with priority queue
        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int cost = curr[0], city = curr[1], stops = curr[2];

            // If we reach the destination, return the cost
            if (city == dst) {
                return cost;
            }

            // If we have reached the max number of stops, continue
            if (stops > k) continue;

            // Explore the neighbors (i.e., flights from the current city)
            if (graph.containsKey(city)) {
                for (int[] neighbor : graph.get(city)) {
                    int nextCity = neighbor[0], price = neighbor[1];
                    int newCost = cost + price;

                    // If the new path is cheaper and hasn't been visited yet at this number of stops
                    if (newCost < visited[nextCity][stops + 1]) {
                        visited[nextCity][stops + 1] = newCost;
                        pq.offer(new int[]{newCost, nextCity, stops + 1});
                    }
                }
            }
        }

        // If we haven't found a path to the destination, return -1
        return -1;
    }

    public static int[] IdentifyingRedundantConnection(int[][] edges) {
        int n = edges.length;  // The number of edges is the same as the number of nodes minus one
        if(n == 1) {
            return new int[0];
        }

        // Union-Find (Disjoint Set) setup
        int[] parent = new int[n + 1]; // parent[i] will store the parent of node i
        int[] rank = new int[n + 1];   // rank[i] will store the rank of node i for efficient union

        // Initialize parent array: initially, each node is its own parent
        for (int i = 1; i <= n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        // Iterate through the edges
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];

            // If u and v are already connected, this edge is the redundant one
            if (find(u, parent) == find(v, parent)) {
                return edge; // Return the redundant edge
            } else {
                union(u, v, parent, rank); // Union the two nodes
            }
        }

        return new int[0];
    }

    public static List<Integer> MinimumHeightTree(int n, int[][] edges) {
        if (n == 1) {
            return Arrays.asList(0);  // If there's only one node, it's the root
        }

        // Step 1: Build the adjacency list from edges
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            adjList.get(edge[0]).add(edge[1]);
            adjList.get(edge[1]).add(edge[0]);
        }

        // Step 2: Find the farthest node from an arbitrary node (e.g., node 0)
        int[] resultFromFirstBFS = bfs(0, n, adjList);
        int farthestNode = resultFromFirstBFS[0];  // Farthest node from node 0

        // Step 3: Find the farthest node from the farthest node (this gives us the diameter)
        int[] resultFromSecondBFS = bfs(farthestNode, n, adjList);
        int otherEnd = resultFromSecondBFS[0];  // Farthest node from 'farthestNode'
        int diameter = resultFromSecondBFS[1];  // The diameter of the tree

        // Step 4: The centers of the tree are at the middle of the path from 'farthestNode' to 'otherEnd'
        List<Integer> centers = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        dist[farthestNode] = 0;
        queue.offer(farthestNode);

        // BFS to find the centers (the middle of the longest path)
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : adjList.get(node)) {
                if (dist[neighbor] == -1) {
                    dist[neighbor] = dist[node] + 1;
                    queue.offer(neighbor);
                }
            }
        }

        // Collect the centers, which are the nodes around the middle of the longest path
        int midDistance = diameter / 2;
        for (int i = 0; i < n; i++) {
            if (dist[i] == midDistance || dist[i] == midDistance + 1) {
                centers.add(i);
            }
        }

        return centers;
    }
    //endregion

    //region Private Methods
    private static int[] bfs(int start, int n, List<List<Integer>> adjList) {
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        dist[start] = 0;

        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);

        int farthestNode = start;
        int maxDist = 0;

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : adjList.get(node)) {
                if (dist[neighbor] == -1) { // not visited
                    dist[neighbor] = dist[node] + 1;
                    queue.offer(neighbor);

                    if (dist[neighbor] > maxDist) {
                        maxDist = dist[neighbor];
                        farthestNode = neighbor;
                    }
                }
            }
        }

        return new int[] {farthestNode, maxDist};  // return the farthest node and its distance
    }

    private static int find(int x, int[] parent) {
        if (parent[x] != x) {
            parent[x] = find(parent[x], parent);  // Path compression
        }
        return parent[x];
    }

    private static void union(int x, int y, int[] parent, int[] rank) {
        int rootX = find(x, parent);
        int rootY = find(y, parent);

        if (rootX != rootY) {
            // Union by rank: attach the smaller tree to the root of the larger tree
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    private static int getLength(ListNode head) {
        int count = 0;
        ListNode curr = head;
        while (curr != null && curr.val != -1) {
            count++;
            curr = curr.next;
        }
        return count;
    }

    public static ListNode buildList(int[] values) {
        if (values.length == 0) {
            return null;
        }
        ListNode head = new ListNode(values[0]);
        ListNode current = head;
        for (int i = 1; i < values.length; i++) {
            current.next = new ListNode(values[i]);
            current = current.next;
        }
        return head;
    }

    private static int zigzagDFS(TreeNode node, boolean moveLeft, int steps) {
        if (node == null) {
            return steps;
        }

        // Move to the opposite child in the next step
        if (moveLeft) {
            return Math.max(zigzagDFS(node.left, false, steps + 1), zigzagDFS(node.right, true, 1));
        } else {
            return Math.max(zigzagDFS(node.right, true, steps + 1), zigzagDFS(node.left, false, 1));
        }
    }

    private static int findLeftDepth(Node node) {
        int depth = 0;
        while (node != null && node.data != -1) {
            node = node.left;
            depth++;
        }
        return depth;
    }

    private static int findRightDepth(Node node) {
        int depth = 0;
        while (node != null && node.data != -1) {
            node = node.right;
            depth++;
        }
        return depth;
    }

    private static List<Integer> getTopView(TreeNode root) {
        if (root == null) return new ArrayList<>();

        // Map to store the first node at each horizontal distance
        Map<Integer, Integer> topViewMap = new TreeMap<>();
        Queue<NodeInfo> queue = new LinkedList<>();
        queue.add(new NodeInfo(root, 0));

        while (!queue.isEmpty()) {
            NodeInfo current = queue.poll();
            TreeNode node = current.node;
            int hd = current.hd;

            // Add the node to the top view if this is the first node at this horizontal distance
            if (!topViewMap.containsKey(hd)) {
                topViewMap.put(hd, node.val);
            }

            // Traverse the left and right children
            if (node.left != null) {
                queue.add(new NodeInfo(node.left, hd - 1));
            }
            if (node.right != null) {
                queue.add(new NodeInfo(node.right, hd + 1));
            }
        }

        // Return the top view values as a list
        return new ArrayList<>(topViewMap.values());
    }

    private static int maxSubArraySum(List<Integer> topView) {
        int maxSum = Integer.MIN_VALUE;
        int currentSum = 0;

        for (int value : topView) {
            currentSum = Math.max(value, currentSum + value);
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum;
    }

    private static String largestNumberFromTree(Node root) {
        // Collect all node values
        List<String> nodeValues = collectNodeValues(root);

        // Sort the node values based on custom comparator
        nodeValues.sort(new LargestNumberComparator());

        // Edge case: If the largest number is 0, the result is "0"
        if (nodeValues.get(0).equals("0")) {
            return "0";
        }

        // Build the largest number by concatenating sorted values
        StringBuilder largestNumber = new StringBuilder();
        for (String value : nodeValues) {
            largestNumber.append(value);
        }

        return largestNumber.toString();
    }

    private static List<String> collectNodeValues(Node root) {
        List<String> values = new ArrayList<>();
        if (root == null) return values;

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            Node current = queue.poll();
            values.add(String.valueOf(current.data));

            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }

        return values;
    }

    private static class LargestNumberComparator implements Comparator<String> {
        @Override
        public int compare(String a, String b) {
            String order1 = a + b;
            String order2 = b + a;
            return order2.compareTo(order1); // sort in descending order
        }
    }

    private static int dfs(Node node) {
        if (node == null) {
            return COVERED_NO_WATCHMAN; // Null nodes are considered covered
        }

        int left = dfs(node.left);
        int right = dfs(node.right);

        // If either child is not covered, we need to place a watchman at this node
        if (left == NOT_COVERED || right == NOT_COVERED) {
            watchmenCount++;
            return HAS_WATCHMAN;
        }

        // If either child has a watchman, this node is covered but doesn't need a watchman
        if (left == HAS_WATCHMAN || right == HAS_WATCHMAN) {
            return COVERED_NO_WATCHMAN;
        }

        // Otherwise, this node is not covered
        return NOT_COVERED;
    }

    private static int inOrderTraversal(TreeNode node, int target, int[] position) {
        if (node == null) return -1;

        // Traverse the left subtree
        int left = inOrderTraversal(node.left, target, position);
        if (left != -1) return left;

        // Visit the current node
        position[0]++;
        if (node.val == target) {
            return position[0]; // Return the 1-based position if target found
        }

        // Traverse the right subtree
        return inOrderTraversal(node.right, target, position);
    }

    private static void precomputeFactorials(int n) {
        fact = new long[n + 1];
        invFact = new long[n + 1];
        fact[0] = 1;
        for (int i = 1; i <= n; i++) {
            fact[i] = fact[i - 1] * i % MOD;
        }
        invFact[n] = modInverse(fact[n]);
        for (int i = n - 1; i >= 0; i--) {
            invFact[i] = invFact[i + 1] * (i + 1) % MOD;
        }
    }

    private static long modInverse(long x) {
        return power(x, AdvancedDataStructure.MOD - 2);
    }

    private static long power(long x, long y) {
        long res = 1;
        while (y > 0) {
            if ((y & 1) == 1) {
                res = res * x % AdvancedDataStructure.MOD;
            }
            x = x * x % AdvancedDataStructure.MOD;
            y >>= 1;
        }
        return res;
    }

    private static int countWays(int[] parks) {
        if (parks.length <= 1) return 1;

        List<Integer> leftSubtree = new ArrayList<>();
        List<Integer> rightSubtree = new ArrayList<>();

        // Split the parks into left and right subtrees based on the root (first element)
        int root = parks[0];
        for (int i = 1; i < parks.length; i++) {
            if (parks[i] < root) {
                leftSubtree.add(parks[i]);
            } else {
                rightSubtree.add(parks[i]);
            }
        }

        // Recursively calculate the number of ways for left and right subtrees
        long leftWays = countWays(toArray(leftSubtree));
        long rightWays = countWays(toArray(rightSubtree));

        // Calculate the number of ways to merge left and right subtrees
        int leftSize = leftSubtree.size();
        int rightSize = rightSubtree.size();
        long mergeWays = binomial(leftSize + rightSize, leftSize);

        // The total ways for this tree is the product of leftWays, rightWays, and mergeWays
        long result = leftWays * rightWays % MOD * mergeWays % MOD;
        return (int) result;
    }

    private static int[] toArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }

    private static long binomial(int n, int k) {
        if (k > n) return 0;
        return fact[n] * invFact[k] % MOD * invFact[n - k] % MOD;
    }

    private static void inOrderTraversal(Node node) {
        if (node == null) {
            return;
        }

        // Traverse the left subtree
        inOrderTraversal(node.left);

        // Process the current node
        if (prevValue != null) {
            // Compute the absolute difference with the previous node
            int diff = Math.abs(node.data - prevValue);
            // Update the minimum difference if necessary
            minDifference = Math.min(minDifference, diff);
        }
        // Update prevValue to the current node value
        prevValue = node.data;

        // Traverse the right subtree
        inOrderTraversal(node.right);
    }

    private static void inOrderTraversal(TreeNode node) {
        if (node == null) return;
        inOrderTraversal(node.left);
        values.add(node.val);
        inOrderTraversal(node.right);
    }

    private static boolean isPossible(int[] books, int n, int students, int maxPages) {
        int studentCount = 1;
        int currentSum = 0;

        for (int pages : books) {
            if (currentSum + pages > maxPages) {
                studentCount++;
                currentSum = pages;

                if (studentCount > students) {
                    return false;
                }
            } else {
                currentSum += pages;
            }
        }

        return true;
    }

    private static int getMax(int[] books) {
        int max = books[0];
        for (int pages : books) {
            max = Math.max(max, pages);
        }
        return max;
    }

    private static int getSum(int[] books) {
        int sum = 0;
        for (int pages : books) {
            sum += pages;
        }
        return sum;
    }

    private static boolean canPlaceCows(int[] stalls, int n, int k, int distance) {
        int count = 1; // Place the first cow in the first stall
        int lastPosition = stalls[0];

        for (int i = 1; i < n; i++) {
            if (stalls[i] - lastPosition >= distance) {
                count++;  // Place a cow in this stall
                lastPosition = stalls[i];

                if (count == k) {
                    return true;  // All cows have been placed successfully
                }
            }
        }

        return false;  // Could not place all cows with at least `distance` separation
    }

    public static Node insertBST(Node root, int value) {
        if (root == null) {
            return new Node(value);
        }
        if (value < root.data) {
            root.left = insertBST(root.left, value);
        } else if (value > root.data) {
            root.right = insertBST(root.right, value);
        }
        return root;
    }

    private static void inOrderTraversal(Node root, List<Integer> sortedList) {
        if (root != null) {
            inOrderTraversal(root.left, sortedList);
            sortedList.add(root.data);
            inOrderTraversal(root.right, sortedList);
        }
    }

    private static SubTreeInfo largestBSTHelper(Node node, int[] maxBSTSize) {
        // Base case
        if (node == null) {
            return new SubTreeInfo(true, 0, Integer.MAX_VALUE, Integer.MIN_VALUE);
        }

        // Recursively check left and right subtrees
        SubTreeInfo leftInfo = largestBSTHelper(node.left, maxBSTSize);
        SubTreeInfo rightInfo = largestBSTHelper(node.right, maxBSTSize);

        // Check if the current subtree rooted at `node` is a BST
        if (leftInfo.isBST && rightInfo.isBST && node.data > leftInfo.max && node.data < rightInfo.min) {
            int size = 1 + leftInfo.size + rightInfo.size;
            maxBSTSize[0] = Math.max(maxBSTSize[0], size);
            int min = Math.min(node.data, leftInfo.min);
            int max = Math.max(node.data, rightInfo.max);
            return new SubTreeInfo(true, size, min, max);
        } else {
            return new SubTreeInfo(false, 0, 0, 0);
        }
    }

    private static void insertIntoTrie(String word, TrieNode root) {
        TrieNode currentNode = root;

        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int index = ch - 'a';

            // If the character node doesn't exist, create it
            if (currentNode.children[index] == null) {
                currentNode.children[index] = new TrieNode(ch);
            }

            // Move to the next node
            currentNode = currentNode.children[index];
        }

        // Mark the end of the word
        currentNode.isend = true;
        System.out.println("Yes");
    }

    private static void insertIntoTrie(int num, TrieNode root) {
        TrieNode current = root;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            if (bit == 0) {
                if (current.left == null) {
                    current.left = new TrieNode();
                }
                current = current.left;
            } else {
                if (current.right == null) {
                    current.right = new TrieNode();
                }
                current = current.right;
            }
        }
    }

    private static void getWordsFromNode(TrieNode node, StringBuilder prefix, List<String> result) {
        if (node.isend) {
            result.add(prefix.toString());
        }
        for (char ch = 'a'; ch <= 'z'; ch++) {
            int index = ch - 'a';
            if (node.children[index] != null) {
                prefix.append(ch);
                getWordsFromNode(node.children[index], prefix, result);
                prefix.deleteCharAt(prefix.length() - 1);  // Backtrack
            }
        }
    }

    private static int findMaxXOR(int num, TrieNode root) {
        TrieNode current = root;
        int maxXOR = 0;

        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            // Check if we can take the opposite bit
            if (bit == 0) {
                if (current.right != null) { // Go right to maximize XOR
                    maxXOR |= (1 << i);
                    current = current.right;
                } else { // Otherwise, go left
                    current = current.left;
                }
            } else {
                if (current.left != null) { // Go left to maximize XOR
                    maxXOR |= (1 << i);
                    current = current.left;
                } else { // Otherwise, go right
                    current = current.right;
                }
            }
        }

        return maxXOR;
    }

    private static boolean removeFromTrieNodeHelper(TrieNode current, String word, int depth) {
        if (current == null) {
            return false;
        }

        // If we've reached the end of the word
        if (depth == word.length()) {
            // Word doesn't exist
            if (!current.isend) {
                return false;
            }
            // Mark the end of word as false
            current.isend = false;

            // If current node has no other children, return true (node can be deleted)
            return isTrieNodeEmpty(current);
        }

        int index = word.charAt(depth) - 'a';
        if (removeFromTrieNodeHelper(current.children[index], word, depth + 1)) {
            current.children[index] = null;

            // Check if current node can be deleted
            return !current.isend && isTrieNodeEmpty(current);
        }
        return false;
    }

    private static boolean isTrieNodeEmpty(TrieNode node) {
        for (TrieNode child : node.children) {
            if (child != null) {
                return false;
            }
        }
        return true;
    }

    private static int countLessEqual(int[][] matrix, int mid) {
        int count = 0;
        int n = matrix.length;
        int row = n - 1; // Start from the bottom left corner
        int col = 0;

        while (row >= 0 && col < n) {
            if (matrix[row][col] <= mid) {
                count += (row + 1);
                col++;
            } else {
                row--;
            }
        }

        return count;
    }

    private static void heapify(int[] arr, int n, int i) {
        int largest = i; // Initialize largest as root
        int left = 2 * i + 1; // Left child index
        int right = 2 * i + 2; // Right child index

        // Check if left child is larger than root
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }

        // Check if right child is larger than the current largest
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }

        // Swap and continue heapifying if root is not the largest
        if (largest != i) {
            int temp = arr[i];
            arr[i] = arr[largest];
            arr[largest] = temp;

            // Recursively heapify the affected subtree
            heapify(arr, n, largest);
        }
    }

    private static int[] dijkstraHelper(int n, Map<Integer, List<int[]>> graph, int source) {
        int[] distances = new int[n];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[source] = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        pq.offer(new int[] { source, 0 }); // {node, distance}

        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int currentNode = current[0];
            int currentDistance = current[1];

            // If the distance in the priority queue is outdated, skip
            if (currentDistance > distances[currentNode]) continue;

            for (int[] neighbor : graph.get(currentNode)) {
                int nextNode = neighbor[0];
                int weight = neighbor[1];
                int newDistance = currentDistance + weight;

                if (newDistance < distances[nextNode]) {
                    distances[nextNode] = newDistance;
                    pq.offer(new int[] { nextNode, newDistance });
                }
            }
        }

        return distances;
    }

    private static boolean hasCycle(List<List<Integer>> graph, int node, int[] visited) {
        visited[node] = 1; // Mark as visiting

        for (int neighbor : graph.get(node)) {
            if (visited[neighbor] == 1) {
                // Found a cycle
                return true;
            }
            if (visited[neighbor] == 0) {
                if (hasCycle(graph, neighbor, visited)) {
                    return true;
                }
            }
        }

        visited[node] = 2; // Mark as visited
        return false;
    }

    private static void dfs(List<List<Integer>> graph, int node, boolean[] visited, Stack<Integer> stack) {
        visited[node] = true;
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfs(graph, neighbor, visited, stack);
            }
        }
        stack.push(node); // Push the node onto the stack after visiting
    }

    private static void dfsTransposed(List<List<Integer>> transposedGraph, int node, boolean[] visited) {
        visited[node] = true;
        for (int neighbor : transposedGraph.get(node)) {
            if (!visited[neighbor]) {
                dfsTransposed(transposedGraph, neighbor, visited);
            }
        }
    }

    public static int minEdgesToAdd(int n, int[][] edges) {
        Graph graph = new Graph(n);

        // Build the graph
        for (int[] edge : edges) {
            graph.addEdge(edge[0] - 1, edge[1] - 1); // Convert to 0-based indexing
        }

        // Find SCCs
        List<List<Integer>> sccs = findSCCs(graph);

        if (sccs.size() == 1) {
            // Already strongly connected
            return 0;
        }

        // Build the condensed graph
        int sccCount = sccs.size();
        boolean[] hasInDegree = new boolean[sccCount];
        boolean[] hasOutDegree = new boolean[sccCount];
        Map<Integer, Integer> nodeToSCC = new HashMap<>();
        for (int i = 0; i < sccs.size(); i++) {
            for (int node : sccs.get(i)) {
                nodeToSCC.put(node, i);
            }
        }

        for (int u = 0; u < n; u++) {
            for (int v : graph.adjList.get(u)) {
                int fromSCC = nodeToSCC.get(u);
                int toSCC = nodeToSCC.get(v);
                if (fromSCC != toSCC) {
                    hasOutDegree[fromSCC] = true;
                    hasInDegree[toSCC] = true;
                }
            }
        }

        // Count SCCs with no in-degree or out-degree
        int inZero = 0, outZero = 0;
        for (int i = 0; i < sccCount; i++) {
            if (!hasInDegree[i]) inZero++;
            if (!hasOutDegree[i]) outZero++;
        }

        // Return the maximum of inZero and outZero
        return Math.max(inZero, outZero);
    }

    private static List<List<Integer>> findSCCs(Graph graph) {
        int vertices = graph.vertices;
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[vertices];

        // Step 1: Perform DFS and store nodes by finish time
        for (int i = 0; i < vertices; i++) {
            if (!visited[i]) {
                dfs(i, visited, stack, graph.adjList);
            }
        }

        // Step 2: Transpose the graph
        List<List<Integer>> transpose = new ArrayList<>();
        for (int i = 0; i < vertices; i++) {
            transpose.add(new ArrayList<>());
        }
        for (int u = 0; u < vertices; u++) {
            for (int v : graph.adjList.get(u)) {
                transpose.get(v).add(u);
            }
        }

        // Step 3: Perform DFS on the transposed graph
        Arrays.fill(visited, false);
        List<List<Integer>> sccs = new ArrayList<>();
        while (!stack.isEmpty()) {
            int node = stack.pop();
            if (!visited[node]) {
                List<Integer> scc = new ArrayList<>();
                dfsOnTranspose(node, visited, transpose, scc);
                sccs.add(scc);
            }
        }

        return sccs;
    }

    private static void dfsOnTranspose(int node, boolean[] visited, List<List<Integer>> transpose, List<Integer> scc) {
        visited[node] = true;
        scc.add(node);
        for (int neighbor : transpose.get(node)) {
            if (!visited[neighbor]) {
                dfsOnTranspose(neighbor, visited, transpose, scc);
            }
        }
    }

    private static void dfs(int node, boolean[] visited, Stack<Integer> stack, List<List<Integer>> graph) {
        visited[node] = true;
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfs(neighbor, visited, stack, graph);
            }
        }
        stack.push(node);
    }

    private static boolean hasCycle(int course, List<List<Integer>> graph, int[] visitStatus) {
        // Mark the course as being visited (currently in the recursion stack)
        visitStatus[course] = 1;

        // Check all prerequisites of the current course
        for (int prereq : graph.get(course)) {
            if (visitStatus[prereq] == 1) {
                // If the prerequisite is already in the recursion stack, a cycle is detected
                return true;
            }
            if (visitStatus[prereq] == 0) {
                // If the prerequisite is unvisited, explore it
                if (hasCycle(prereq, graph, visitStatus)) {
                    return true;
                }
            }
        }

        // Mark the course as fully visited (exit the recursion stack)
        visitStatus[course] = 2;
        return false;
    }

    private static void dfs(int[][] heights, boolean[][] reach, int i, int j) {
        int m = heights.length;
        int n = heights[0].length;

        // If the cell is already visited, return
        if (reach[i][j]) return;

        // Mark the cell as reachable
        reach[i][j] = true;

        // Explore the neighboring cells (up, down, left, right)
        int[] dirs = {-1, 0, 1, 0, -1}; // Directions: up, right, down, left
        for (int d = 0; d < 4; d++) {
            int ni = i + dirs[d];
            int nj = j + dirs[d + 1];

            if (ni >= 0 && nj >= 0 && ni < m && nj < n && heights[ni][nj] >= heights[i][j]) {
                dfs(heights, reach, ni, nj);
            }
        }
    }
    //endregion
}
