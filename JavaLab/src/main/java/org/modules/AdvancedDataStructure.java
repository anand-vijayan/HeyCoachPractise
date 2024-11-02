package org.modules;

import org.dto.*;

import java.util.*;

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
    //endregion

    //region Heaps 2
    //endregion

    //region Heaps 3
    //endregion

    //region Graphs 1
    //endregion

    //region Graphs 2
    //endregion

    //region Graphs 3
    //endregion

    //region Private Methods
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
    //endregion
}
