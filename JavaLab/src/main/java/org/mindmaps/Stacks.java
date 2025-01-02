package org.mindmaps;

import org.dto.ListNode;
import org.dto.Node;
import org.dto.TreeNode;
import org.helpers.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class Stacks {
    //region Variables & Constants
    private static TreeNode prev=null, head=null;
    private static int index = 0;
    //endregion

    //region Simple Stack
    public static int MaximumNestingDepthOfTheParentheses(String s) {
        int currentDepth = 0;
        int maxDepth = 0;

        for (char c : s.toCharArray()) {
            if (c == '(') {
                currentDepth++;
                maxDepth = Math.max(maxDepth, currentDepth);
            } else if (c == ')') {
                currentDepth--;
            }
        }

        return maxDepth;
    }

    public static String RemoveOutermostParentheses(String s) {
        StringBuilder result = new StringBuilder();
        int depth = 0;

        for (char c : s.toCharArray()) {
            if (c == '(') {
                // If depth is greater than 0, it's not an outermost parenthesis
                if (depth > 0) {
                    result.append(c);
                }
                depth++;
            } else if (c == ')') {
                depth--;
                // If depth is still greater than 0, it's not an outermost parenthesis
                if (depth > 0) {
                    result.append(c);
                }
            }
        }

        return result.toString();
    }

    public static int BaseBallGame(String[] operations) {
        Stack<Integer> record = new Stack<>();

        for (String op : operations) {
            switch (op) {
                case "+" -> {
                    // Add the last two scores
                    int top = record.pop();
                    int newScore = top + record.peek();
                    record.push(top); // Push back the popped element

                    record.push(newScore);
                }
                case "D" ->
                    // Double the last score
                        record.push(2 * record.peek());
                case "C" ->
                    // Remove the last score
                        record.pop();
                default ->
                    // Record a new integer score
                        record.push(Integer.parseInt(op));
            }
        }

        // Calculate the total score
        int totalScore = 0;
        for (int score : record) {
            totalScore += score;
        }

        return totalScore;
    }

    public static String RemoveAllAdjacentDuplicatesInString(String s) {
        StringBuilder stack = new StringBuilder();

        for (char c : s.toCharArray()) {
            if (!stack.isEmpty() && stack.charAt(stack.length() - 1) == c) {
                // Remove the last character if it's the same as the current one
                stack.deleteCharAt(stack.length() - 1);
            } else {
                // Otherwise, add the current character to the stack
                stack.append(c);
            }
        }

        return stack.toString();
    }

    public static List<String> BuildAnArrayWithStackOperations(int[] target, int n) {
        List<String> operations = new ArrayList<>();
        int j = 0; // Pointer for the target array

        for (int i = 1; i <= n; i++) {
            if (j >= target.length) break; // Stop if the target is fully matched

            if (i == target[j]) {
                operations.add("Push");
                j++; // Move to the next target element
            } else {
                operations.add("Push");
                operations.add("Pop"); // Discard the current number
            }
        }

        return operations;
    }
    //endregion

    //region Monotonic Stack
    public static int[] FinalPricesWithASpecialDiscountInAShop(int[] prices) {
        int n = prices.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < n; i++) {
            // Find discounts for items in the stack
            while (!stack.isEmpty() && prices[stack.peek()] >= prices[i]) {
                int idx = stack.pop();
                result[idx] = prices[idx] - prices[i];
            }
            // Push the current index onto the stack
            stack.push(i);
        }

        // Process items without discounts
        while (!stack.isEmpty()) {
            int idx = stack.pop();
            result[idx] = prices[idx];
        }

        return result;
    }

    public static int[] NextGreaterElement1(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> nextGreaterMap = new HashMap<>();
        Stack<Integer> stack = new Stack<>();

        // Precompute the next greater elements for nums2
        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums2[i]) {
                stack.pop();
            }
            nextGreaterMap.put(nums2[i], stack.isEmpty() ? -1 : stack.peek());
            stack.push(nums2[i]);
        }

        // Build the result for nums1
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = nextGreaterMap.get(nums1[i]);
        }

        return result;
    }

    public static void MinStack1() {
        MinStack minStack = new MinStack();
        minStack.push(-2);
        minStack.push(0);
        minStack.push(-3);
        System.out.println(minStack.getMin()); // Output: -3
        minStack.pop();
        System.out.println(minStack.top());    // Output: 0
        System.out.println(minStack.getMin()); // Output: -2
    }

    public static TreeNode MaximumBinaryTree1(int[] nums) {
        return buildTree(nums, 0, nums.length - 1);
    }

    public static int[] DailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] answer = new int[n];
        Stack<Integer> stack = new Stack<>();

        // Iterate through the temperatures array
        for (int i = 0; i < n; i++) {
            // Check if the current temperature is greater than the temperature at the top of the stack
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int index = stack.pop();
                answer[index] = i - index;  // The number of days to wait
            }
            // Push the current index onto the stack
            stack.push(i);
        }

        return answer;
    }

    //endregion

    //region Two Pointer
    public static int MaximumTwinSumOfALinkedList1(ListNode head) {
        if (head == null || head.next == null) return 0;

        // Step 1: Find the middle of the linked list
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // Step 2: Reverse the second half of the list
        ListNode prev = null, curr = slow;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        // Step 3: Calculate twin sums and find the maximum
        int maxSum = 0;
        ListNode firstHalf = head, secondHalf = prev;
        while (secondHalf != null) {
            maxSum = Math.max(maxSum, firstHalf.val + secondHalf.val);
            firstHalf = firstHalf.next;
            secondHalf = secondHalf.next;
        }

        return maxSum;
    }

    public static int MinimumNumberOfSwapsToMakeTheStringBalanced(String s) {
        int balance = 0;
        int maxImbalance = 0;

        // Traverse the string to calculate the maximum imbalance
        for (char c : s.toCharArray()) {
            if (c == '[') {
                balance++;
            } else {
                balance--;
            }
            // Track the maximum imbalance (negative balance)
            if (balance < 0) {
                maxImbalance = Math.max(maxImbalance, -balance);
            }
        }

        // The minimum number of swaps needed is maxImbalance / 2
        return (maxImbalance + 1) / 2;
    }

    public static int TrappingRainWater1(int[] height) {
        if (height == null || height.length == 0) return 0;

        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int trappedWater = 0;

        while (left <= right) {
            if (height[left] <= height[right]) {
                if (height[left] < leftMax) {
                    trappedWater += leftMax - height[left];
                } else {
                    leftMax = height[left];
                }
                left++;
            } else {
                if (height[right] < rightMax) {
                    trappedWater += rightMax - height[right];
                } else {
                    rightMax = height[right];
                }
                right--;
            }
        }

        return trappedWater;
    }

    public static int MaximumScoreOfAGoodSubArray(int[] nums, int k) {
        int n = nums.length;
        int left = k, right = k;
        int min = nums[k];
        int maxScore = nums[k]; // Initialize with the score of the subarray containing only nums[k]

        while (left > 0 || right < n - 1) {
            // Expand to the side that gives a larger minimum value
            if (left > 0 && (right == n - 1 || nums[left - 1] >= nums[right + 1])) {
                left--; // Expand left
                min = Math.min(min, nums[left]);
            } else if (right < n - 1) {
                right++; // Expand right
                min = Math.min(min, nums[right]);
            }
            // Calculate the score for the current subarray
            maxScore = Math.max(maxScore, min * (right - left + 1));
        }

        return maxScore;
    }
    //endregion

    //region DP Based
    public static int LongestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        stack.push(-1); // Initialize with -1 to handle the base case
        int maxLength = 0;

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i); // Push the index of '('
            } else {
                stack.pop(); // Pop for ')'
                if (stack.isEmpty()) {
                    stack.push(i); // Update the base for the next valid substring
                } else {
                    maxLength = Math.max(maxLength, i - stack.peek());
                }
            }
        }

        return maxLength;
    }

    public static int TrappingRainWater2(int[] height) {
        return TrappingRainWater1(height);
    }

    public static int MaximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int rows = matrix.length;
        int cols = matrix[0].length;
        int[] heights = new int[cols];
        int maxArea = 0;

        for (char[] chars : matrix) {
            // Update the heights array
            for (int j = 0; j < cols; j++) {
                if (chars[j] == '1') {
                    heights[j]++;
                } else {
                    heights[j] = 0;
                }
            }

            // Calculate the maximum area for the current histogram
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }

        return maxArea;
    }

    public static int MinimumNumberOfIncrementsOnSubArraysToFormATargetArray(int[] target) {
        int operations = target[0]; // Initialize with the first element
        for (int i = 1; i < target.length; i++) {
            if (target[i] > target[i - 1]) {
                operations += target[i] - target[i - 1];
            }
        }
        return operations;
    }

    public static boolean ValidParenthesisString(String s) {
        int low = 0, high = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                low++;
                high++;
            } else if (c == ')') {
                low = Math.max(low - 1, 0);
                high--;
            } else if (c == '*') {
                low = Math.max(low - 1, 0); // Treat '*' as ')' or empty
                high++; // Treat '*' as '('
            }
            if (high < 0) {
                return false; // Too many ')' at any point
            }
        }
        return low == 0; // All unmatched '(' are balanced
    }
    //endregion

    //region Recursion
    public static List<Integer> NaryTreePostOrderTraversal(Node root) {
        List<Integer> result = new ArrayList<>();
        postorderHelper(root, result);
        return result;
    }

    public static List<Integer> NaryTreePreOrderTraversal(Node root) {
        List<Integer> result = new ArrayList<>();
        preorderHelper(root, result);
        return result;
    }

    public static int MaximumTwinSumOfALinkedList2(ListNode head) {
        return MaximumTwinSumOfALinkedList1(head);
    }

    public static int BasicCalculator(String s) {
        // Initialize a stack to store numbers and signs
        Stack<Integer> stack = new Stack<>();
        int currentNumber = 0;
        int result = 0; // The final result
        int sign = 1;   // 1 means positive, -1 means negative

        // Iterate over the string
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);

            // If the character is a digit, build the current number
            if (Character.isDigit(ch)) {
                currentNumber = currentNumber * 10 + (ch - '0');
            }

            // If the character is a '+', add the current number to result
            if (ch == '+') {
                result += sign * currentNumber;
                sign = 1; // Set sign to positive for next number
                currentNumber = 0;
            }

            // If the character is a '-', subtract the current number from result
            if (ch == '-') {
                result += sign * currentNumber;
                sign = -1; // Set sign to negative for next number
                currentNumber = 0;
            }

            // If the character is an open parenthesis, push the result and sign onto the stack
            if (ch == '(') {
                stack.push(result);
                stack.push(sign);
                result = 0;
                sign = 1;
            }

            // If the character is a closing parenthesis, pop the sign and the result
            if (ch == ')') {
                result += sign * currentNumber;
                currentNumber = 0;
                result *= stack.pop(); // Pop the sign
                result += stack.pop(); // Pop the result before the parenthesis
            }
        }

        // Add the last number to the result
        result += sign * currentNumber;

        return result;
    }
    //endregion

    //region Design
    public static void ImplementQueueUsingStacks() {
        MyQueue queue = new MyQueue();

        // Test case
        queue.push(1);
        queue.push(2);
        System.out.println(queue.peek()); // Output: 1
        System.out.println(queue.pop());  // Output: 1
        System.out.println(queue.empty()); // Output: false
        System.out.println(queue.pop());  // Output: 2
        System.out.println(queue.empty()); // Output: true
    }

    public static void ImplementStackUsingQueues(){
        MyStack stack = new MyStack();

        // Test case
        stack.push(1);
        stack.push(2);
        System.out.println(stack.top());  // Output: 2
        System.out.println(stack.pop());  // Output: 2
        System.out.println(stack.empty()); // Output: false
        System.out.println(stack.pop());  // Output: 1
        System.out.println(stack.empty()); // Output: true
    }

    public static void MinStack2() {
        MinStack1();
    }

    public static void DesignAStackWithIncrementOperation() {
        CustomStack customStack = new CustomStack(3);

        customStack.push(1);  // stack = [1]
        customStack.push(2);  // stack = [1, 2]
        customStack.push(3);  // stack = [1, 2, 3]
        System.out.println(customStack.pop());   // Output: 3, stack = [1, 2]
        customStack.push(4);  // stack = [1, 2, 4]
        customStack.increment(2, 5);  // Increment the first 2 elements by 5, stack = [6, 7, 4]
        System.out.println(customStack.pop());   // Output: 7, stack = [6, 4]
        System.out.println(customStack.pop());   // Output: 6, stack = [4]
        System.out.println(customStack.pop());   // Output: 4, stack = []
        System.out.println(customStack.pop());   // Output: -1, stack is empty
    }

    public static void OnlineStockSpan(){
        StockSpanner stockSpanner = new StockSpanner();
        System.out.println(stockSpanner.next(100)); // 1
        System.out.println(stockSpanner.next(80));  // 1
        System.out.println(stockSpanner.next(60));  // 1
        System.out.println(stockSpanner.next(70));  // 2
        System.out.println(stockSpanner.next(60));  // 1
        System.out.println(stockSpanner.next(75));  // 4
        System.out.println(stockSpanner.next(85));  // 6
    }
    //endregion

    //region Binary Tree Based
    public static TreeNode IncreasingOrderSearchTree(TreeNode root){
        if(root==null) return null;
        IncreasingOrderSearchTree(root.left);
        if(prev!=null) {
            root.left=null; // we no  longer needs the left  side of the node, so set it to null
            prev.right=root;
        }
        if(head==null) head=root; // record the most left node as it will be our root
        prev=root; //keep track of the prev node
        IncreasingOrderSearchTree(root.right);
        return head;
    }

    public static List<Integer> BinaryTreeInorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;

        // Traverse the tree iteratively using a stack
        while (current != null || !stack.isEmpty()) {
            // Reach the leftmost node of the current node
            while (current != null) {
                stack.push(current);
                current = current.left;
            }

            // Pop the top item from the stack and add its value to the result
            current = stack.pop();
            result.add(current.val);

            // Move to the right subtree
            current = current.right;
        }

        return result;
    }

    public static List<Integer> BinaryTreePostOrderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        btPostOrder(root, result);
        return result;
    }

    public static List<Integer> BinaryTreePreOrderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        btPreOrder(root, result);
        return result;
    }

    public static boolean PalindromeLinkedList(ListNode head) {
        ListNode slow = head, fast = head, prev, temp;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        prev = slow;
        slow = slow.next;
        prev.next = null;
        while (slow != null) {
            temp = slow.next;
            slow.next = prev;
            prev = slow;
            slow = temp;
        }
        fast = head;
        slow = prev;
        while (slow != null) {
            if (fast.val != slow.val) return false;
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }
    //endregion

    //region Hashing
    public static int[] NextGreaterElement2(int[] nums1, int[] nums2) {
        return NextGreaterElement1(nums1,nums2);
    }

    public static TreeNode ConstructBinarySearchTreeFromPreorderTraversal1(int[] preorder) {
        return buildTree(preorder, Integer.MAX_VALUE);
    }
    //endregion

    //region Divide And Conquer
    public static TreeNode MaximumBinaryTree2(int[] nums) {
        return MaximumBinaryTree1(nums);
    }

    public static TreeNode ConstructBinarySearchTreeFromPreorderTraversal2(int[] preorder){
        return ConstructBinarySearchTreeFromPreorderTraversal1(preorder);
    }
    //endregion

    //region Private Method
    private static TreeNode buildTree(int[] nums, int left, int right) {
        // Base case: if the subarray is empty, return null
        if (left > right) {
            return null;
        }

        // Find the index of the maximum element in the current subarray
        int maxIndex = findMaxIndex(nums, left, right);

        // Create the root node with the maximum value
        TreeNode root = new TreeNode(nums[maxIndex]);

        // Recursively build the left and right subtrees
        root.left = buildTree(nums, left, maxIndex - 1);
        root.right = buildTree(nums, maxIndex + 1, right);

        return root;
    }

    private static int findMaxIndex(int[] nums, int left, int right) {
        int maxIndex = left;
        for (int i = left + 1; i <= right; i++) {
            if (nums[i] > nums[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int n = heights.length;

        for (int i = 0; i <= n; i++) {
            int currentHeight = (i == n) ? 0 : heights[i];

            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }

            stack.push(i);
        }

        return maxArea;
    }

    private static void postorderHelper(Node node, List<Integer> result) {
        if (node == null) {
            return;
        }

        // Traverse all children first
        for (Node child : node.children) {
            postorderHelper(child, result);
        }

        // After visiting children, visit the current node
        result.add(node.value);
    }

    private static void preorderHelper(Node node, List<Integer> result) {
        if (node == null) {
            return;
        }

        // Visit the current node
        result.add(node.value);

        // Traverse all children
        for (Node child : node.children) {
            preorderHelper(child, result);
        }
    }

    private static void btPostOrder(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;  // Base case: if the node is null, just return
        }

        btPostOrder(node.left, result);  // Traverse left subtree
        btPostOrder(node.right, result); // Traverse right subtree
        result.add(node.val);         // Visit current node
    }

    private static void btPreOrder(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;  // Base case: if the node is null, just return
        }

        result.add(node.val);         // Visit current node
        btPreOrder(node.left, result);  // Traverse left subtree
        btPreOrder(node.right, result); // Traverse right subtree
    }

    private static TreeNode buildTree(int[] preorder, int bound) {
        // Base case: If all elements are used or the current element is larger than the bound
        if (index == preorder.length || preorder[index] > bound) {
            return null;
        }

        // Create the node for the current root
        TreeNode root = new TreeNode(preorder[index++]);

        // Recursively build the left subtree (values < root)
        root.left = buildTree(preorder, root.val);

        // Recursively build the right subtree (values > root)
        root.right = buildTree(preorder, bound);

        return root;
    }
    //endregion
}
