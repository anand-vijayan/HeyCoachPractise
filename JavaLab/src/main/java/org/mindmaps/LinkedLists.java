package org.mindmaps;

import org.dto.ListNode;
import java.util.*;

public class LinkedLists {

    //region Sorting
    public static ListNode RemoveDuplicatesFromSortedList1(ListNode head) {
        if (head == null) {
            return null; // If the list is empty, return null
        }

        ListNode current = head;

        // Traverse the list
        while (current.next != null) {
            // If the current node's value is the same as the next node's value
            if (current.val == current.next.val) {
                // Skip the next node by changing the current node's next pointer
                current.next = current.next.next;
            } else {
                // Otherwise, just move to the next node
                current = current.next;
            }
        }

        return head; // Return the modified list
    }

    public static ListNode RemoveDuplicatesFromSortedList2(ListNode head) {
        // Create a fake node to handle edge cases easily (like removing the first node)
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;

        while (head != null) {
            // Check if the current node is a duplicate (i.e., the next node has the same value)
            if (head.next != null && head.val == head.next.val) {
                // Skip all nodes with the same value
                while (head.next != null && head.val == head.next.val) {
                    head = head.next;
                }
                // Move the prev node's next pointer to the node after the last duplicate
                prev.next = head.next;
            } else {
                // Otherwise, move prev to the current node
                prev = prev.next;
            }
            // Move head forward
            head = head.next;
        }

        // Return the modified list starting from the next node of fake
        return dummy.next;
    }

    public static ListNode SortList1(ListNode head) {
        if (head == null || head.next == null) {
            return head; // Base case: A list of length 0 or 1 is already sorted.
        }

        // Step 1: Split the list into two halves using the fast and slow pointer technique
        ListNode middle = getMiddle(head);
        ListNode right = middle.next;
        middle.next = null;

        // Step 2: Recursively sort each half
        ListNode left = SortList1(head);
        right = SortList1(right);

        // Step 3: Merge the sorted halves
        return merge(left, right);
    }
    //endregion

    //region Rotating
    public static ListNode RotateListRight(ListNode head, int k) {
        if (head == null || head.next == null || k == 0) {
            return head; // Edge cases: empty list, single node, or no rotation needed
        }

        // Step 1: Calculate the length of the list
        ListNode current = head;
        int length = 1; // Start counting from the head
        while (current.next != null) {
            current = current.next;
            length++;
        }

        // Step 2: Normalize k
        k %= length;
        if (k == 0) {
            return head; // If k is a multiple of the length, no rotation is needed
        }

        // Step 3: Find the new tail
        int newTailPosition = length - k - 1;
        current = head;
        for (int i = 0; i < newTailPosition; i++) {
            current = current.next;
        }
        ListNode newTail = current;
        ListNode newHead = newTail.next;

        // Step 4: Perform the rotation
        newTail.next = null; // Break the list
        current = newHead; // Start from the new head
        while (current.next != null) {
            current = current.next; // Move to the end of the new list
        }
        current.next = head; // Link the old head to the end of the new list

        return newHead;
    }
    //endregion

    //region Reversing
    public static ListNode ReverseLinkedList1(ListNode head) {
        ListNode prev = null; // To store the previous node
        ListNode current = head; // To traverse the list

        while (current != null) {
            ListNode next = current.next; // Store the next node
            current.next = prev;         // Reverse the link
            prev = current;              // Move prev to the current node
            current = next;              // Move to the next node
        }

        return prev; // New head of the reversed list
    }

    public static ListNode ReverseLinkedList2(ListNode head, int left, int right) {
        if (head == null || left == right) {
            return head; // No reversal needed
        }

        // Step 1: Create a dummy node to handle edge cases like reversing from the head
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;

        // Step 2: Move `prev` to the node just before position `left`
        for (int i = 1; i < left; i++) {
            prev = prev.next;
        }

        // Step 3: Reverse the sublist from `left` to `right`
        ListNode curr = prev.next; // Start of the sublist
        ListNode next;
        ListNode tail = curr; // The node at `left` becomes the tail of the reversed sublist

        ListNode reversedSublist = null; // Reversed part of the sublist
        for (int i = 0; i <= right - left; i++) {
            next = curr.next;      // Temporarily store the next node
            curr.next = reversedSublist; // Reverse the link
            reversedSublist = curr; // Move reversedSublist to the current node
            curr = next;           // Move to the next node
        }

        // Step 4: Reconnect the reversed sublist
        prev.next = reversedSublist; // Connect the node before `left` to the head of the reversed sublist
        tail.next = curr;           // Connect the tail of the reversed sublist to the node after `right`

        return dummy.next; // Return the new head
    }
    //endregion

    //region Splitting
    public static ListNode[] SplitLinkedListToParts(ListNode head, int k) {
        ListNode[] result = new ListNode[k];

        // Step 1: Calculate the length of the list
        int length = 0;
        ListNode current = head;
        while (current != null) {
            length++;
            current = current.next;
        }

        // Step 2: Determine the size of each part
        int partSize = length / k;
        int remainder = length % k; // Extra nodes to distribute

        // Step 3: Split the list
        current = head;
        for (int i = 0; i < k; i++) {
            result[i] = current; // Assign the start of the current part
            int currentPartSize = partSize + (i < remainder ? 1 : 0); // Add an extra node for the first `remainder` parts

            for (int j = 0; j < currentPartSize - 1; j++) {
                if (current != null) {
                    current = current.next;
                }
            }

            // Disconnect the current part from the rest of the list
            if (current != null) {
                ListNode temp = current.next;
                current.next = null;
                current = temp;
            }
        }

        return result;
    }
    //endregion

    //region Removing
    public static ListNode RemoveLinkedListElements(ListNode head, int val) {
        // Step 1: Create a dummy node
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode current = dummy;

        // Step 2: Traverse the list and remove nodes with value `val`
        while (current.next != null) {
            if (current.next.val == val) {
                current.next = current.next.next; // Skip the node with value `val`
            } else {
                current = current.next; // Move to the next node
            }
        }

        // Step 3: Return the new head
        return dummy.next;
    }

    public static ListNode RemoveNthNodeFromEndOfList(ListNode head, int n) {
        // Step 1: Create a dummy node
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;

        // Step 2: Move `fast` pointer `n + 1` steps ahead
        for (int i = 0; i <= n; i++) {
            fast = fast.next;
        }

        // Step 3: Move both `fast` and `slow` until `fast` reaches the end
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }

        // Step 4: Remove the target node
        slow.next = slow.next.next;

        // Step 5: Return the new head
        return dummy.next;
    }

    public static ListNode RemoveDuplicatesFromLinkedList2(ListNode head) {
        return RemoveDuplicatesFromSortedList2(head);
    }

    public static ListNode RemoveDuplicatesFromLinkedList1(ListNode head) {
        return RemoveDuplicatesFromSortedList1(head);
    }

    public static void DeleteNodeInALinkedList(ListNode node){
        // Copy the value of the next node to the current node
        node.val = node.next.val;

        // Update the next pointer of the current node to skip the next node
        node.next = node.next.next;
    }

    //endregion

    //region Re-ordering
    public static void ReorderLinkedList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }

        // Step 1: Find the middle of the list using slow and fast pointers
        ListNode slow = head;
        ListNode fast = head;

        // Using the fast and slow pointer technique to find the middle
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // Step 2: Reverse the second half of the list
        ListNode second = slow.next;
        slow.next = null;  // Split the list into two halves
        second = reverse(second);

        // Step 3: Merge the two halves
        ListNode first = head;
        while (second != null) {
            ListNode tmp1 = first.next;
            ListNode tmp2 = second.next;

            first.next = second;  // Connect first half to second half
            second.next = tmp1;   // Connect second half to next node in first half
            first = tmp1;         // Move first pointer to next node in first half
            second = tmp2;        // Move second pointer to next node in second half
        }
    }

    public static ListNode OddEvenLinkedList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        // Initialize odd and even pointers
        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even; // To remember the start of even list

        // Traverse the list and separate odd and even nodes
        while (even != null && even.next != null) {
            odd.next = odd.next.next;  // Connect odd nodes
            even.next = even.next.next; // Connect even nodes

            odd = odd.next;  // Move odd pointer
            even = even.next;  // Move even pointer
        }

        // Connect the end of odd list to the start of even list
        odd.next = evenHead;

        return head;
    }
    //endregion

    //region Multiple Pointer
    public static ListNode MiddleOfTheLinkedList(ListNode head) {
        // Initialize slow and fast pointers
        ListNode slow = head;
        ListNode fast = head;

        // Move fast pointer by 2 steps and slow pointer by 1 step
        while (fast != null && fast.next != null) {
            slow = slow.next;        // Move slow pointer 1 step
            fast = fast.next.next;   // Move fast pointer 2 steps
        }

        // When fast reaches the end, slow will be at the middle
        return slow;
    }

    public static ListNode DeleteTheMiddleNodeOfALinkedList(ListNode head) {
        // If the list has only one node, return null
        if (head == null || head.next == null) {
            return null;
        }

        // Initialize slow and fast pointers
        ListNode slow = head;
        ListNode fast = head;
        ListNode prev = null; // To track the node before the slow pointer

        // Traverse the list with slow and fast pointers
        while (fast != null && fast.next != null) {
            fast = fast.next.next;  // Fast pointer moves 2 steps
            prev = slow;            // Store the previous node of slow
            slow = slow.next;       // Slow pointer moves 1 step
        }

        // Delete the middle node (slow is at the middle node now)
        prev.next = slow.next;

        return head;
    }

    public static void ReorderList(ListNode head){
        ReorderLinkedList(head);
    }

    public static ListNode RemoveNthFromEnd(ListNode head, int n) {
        return RemoveNthNodeFromEndOfList(head,n);
    }

    public static ListNode PartitionLinkedList(ListNode head, int x) {
        // Create two dummy nodes for the two lists
        ListNode smallerHead = new ListNode(0); // Head of the smaller list (values < x)
        ListNode greaterHead = new ListNode(0); // Head of the greater list (values >= x)

        // Pointers for the two lists
        ListNode smaller = smallerHead;
        ListNode greater = greaterHead;

        // Traverse the original list
        while (head != null) {
            if (head.val < x) {
                smaller.next = head;  // Add node to smaller list
                smaller = smaller.next;
            } else {
                greater.next = head;  // Add node to greater list
                greater = greater.next;
            }
            head = head.next;
        }

        // Connect the two lists
        greater.next = null;  // Ensure the last node of the greater list points to null
        smaller.next = greaterHead.next;  // Link smaller list to greater list

        // Return the merged list starting from the next node of smallerHead (skipping dummy)
        return smallerHead.next;
    }

    //endregion

    //region Recursion
    public static ListNode AddTwoNumbers1(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0); // Dummy node to simplify handling the head of the result list
        ListNode current = dummyHead; // Pointer to build the result list
        int carry = 0; // Carry value for the addition

        // Traverse both lists until both are null
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry; // Start with the carry from the previous iteration

            if (l1 != null) {
                sum += l1.val; // Add value from the first list
                l1 = l1.next;  // Move to the next node in l1
            }

            if (l2 != null) {
                sum += l2.val; // Add value from the second list
                l2 = l2.next;  // Move to the next node in l2
            }

            carry = sum / 10; // Update carry (value greater than or equal to 10 will carry over)
            current.next = new ListNode(sum % 10); // Create a new node for the current digit
            current = current.next; // Move to the next node in the result list
        }

        return dummyHead.next; // Return the head of the result list
    }

    public static ListNode MergeTwoSortedLists(ListNode list1, ListNode list2) {
        // Dummy node to simplify the merging process
        ListNode dummyHead = new ListNode(0);
        ListNode current = dummyHead; // Pointer to build the merged list

        // Merge the two lists
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                current.next = list1; // Append the smaller node to the merged list
                list1 = list1.next;   // Move the pointer in list1
            } else {
                current.next = list2; // Append the smaller node to the merged list
                list2 = list2.next;   // Move the pointer in list2
            }
            current = current.next; // Move the pointer in the merged list
        }

        // If one list is not empty, append the remaining part of it
        if (list1 != null) {
            current.next = list1;
        } else if (list2 != null) {
            current.next = list2;
        }

        return dummyHead.next;
    }

    public static int ConvertBinaryNumberInALinkedListToInteger(ListNode head) {
        int result = 0;

        // Traverse the linked list
        while (head != null) {
            result = result * 2 + head.val; // Shift left and add the current node's value
            head = head.next; // Move to the next node
        }

        return result; // Return the final decimal value
    }

    public static ListNode SwapNodesInPairs(ListNode head) {
        // Dummy node before the head to simplify edge cases
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode current = dummy;

        // Traverse the list in pairs
        while (current.next != null && current.next.next != null) {
            ListNode node1 = current.next;
            ListNode node2 = current.next.next;

            // Swap the nodes
            node1.next = node2.next;
            node2.next = node1;
            current.next = node2;

            // Move current two nodes ahead
            current = node1;
        }

        return dummy.next; // Return the new head (dummy.next)
    }

    public static ListNode RemoveDuplicatesFromSortedList3(ListNode head){
        return RemoveDuplicatesFromLinkedList1(head);
    }

    //endregion

    //region Iterative Manner
    public static ListNode AddTwoNumbers2(ListNode l1, ListNode l2){
        return AddTwoNumbers1(l1,l2);
    }

    public static ListNode MergeTwoSortedListsInIterativeManner(ListNode list1, ListNode list2){
        return MergeTwoSortedLists(list1, list2);
    }

    public static int ConvertBinaryNumberInALinkedListToIntegerInIterativeManner(ListNode head){
        return ConvertBinaryNumberInALinkedListToInteger(head);
    }

    public static ListNode MiddleOfTheLinkedListInIterativeManner(ListNode head) {
        return MiddleOfTheLinkedList(head);
    }

    public static ListNode DeleteTheMiddleNodeOfALinkedListInIterativeManner(ListNode head){
        return DeleteTheMiddleNodeOfALinkedList(head);
    }

    //endregion

    //region Stack-Based Questions
    public static void ReorderList2(ListNode head){
        ReorderLinkedList(head);
    }

    public static ListNode AddTwoNumbers3(ListNode l1, ListNode l2) {
        // Use two stacks to hold the digits of each number
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();

        // Push all digits of l1 into stack1
        while (l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }

        // Push all digits of l2 into stack2
        while (l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }

        ListNode result = null; // To build the result linked list
        int carry = 0;

        // While there are digits in either stack or a carry left
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
            int sum = carry; // Start with carry value

            // Add digit from stack1 if available
            if (!stack1.isEmpty()) {
                sum += stack1.pop();
            }

            // Add digit from stack2 if available
            if (!stack2.isEmpty()) {
                sum += stack2.pop();
            }

            // Update carry for the next iteration
            carry = sum / 10;

            // Create a new node with the current digit and add it to the result list
            ListNode node = new ListNode(sum % 10);
            node.next = result;
            result = node; // Move the result pointer to the new node
        }

        return result; // Return the head of the result list
    }

    public static int[] NextGreaterNodeInLinkedList(ListNode head) {
        // Step 1: Traverse the linked list and convert it to an array of values
        ListNode current = head;
        ArrayList<Integer> values = new ArrayList<>();

        while (current != null) {
            values.add(current.val);
            current = current.next;
        }

        // Step 2: Initialize the result array
        int n = values.size();
        int[] result = new int[n];

        // Step 3: Use a stack to keep track of indices
        Stack<Integer> stack = new Stack<>();

        // Step 4: Traverse the array of values
        for (int i = 0; i < n; i++) {
            // While the stack is not empty and the current value is greater than
            // the value at the index stored at the top of the stack
            while (!stack.isEmpty() && values.get(i) > values.get(stack.peek())) {
                int index = stack.pop(); // Get the index of the node for which we found the next greater value
                result[index] = values.get(i); // Set the next greater value for this node
            }

            // Push the current index to the stack
            stack.push(i);
        }

        // Step 5: Return the result array
        return result;
    }
    //endregion

    //region Divide and Conquer
    public static ListNode SortList2(ListNode head){
        return SortList1(head);
    }
    //endregion

    //region Hashing
    public static ListNode IntersectionOfTwoLinkedLists(ListNode headA, ListNode headB) {
        // Base case: If either list is empty, return null
        if (headA == null || headB == null) {
            return null;
        }

        ListNode pointerA = headA;
        ListNode pointerB = headB;

        // Traverse both lists
        while (pointerA != pointerB) {
            // If we reach the end of a list, move to the start of the other list
            pointerA = (pointerA == null) ? headB : pointerA.next;
            pointerB = (pointerB == null) ? headA : pointerB.next;
        }

        // Either both are null (no intersection) or they meet at the intersection node
        return pointerA;
    }

    public static int LinkedListComponents(ListNode head, int[] numbers) {
        // Step 1: Store all the values in numbers in a HashSet for quick lookup
        Set<Integer> set = new HashSet<>();
        for (int num : numbers) {
            set.add(num);
        }

        // Step 2: Initialize a counter for the connected components
        int count = 0;
        boolean inComponent = false;

        // Step 3: Traverse the linked list
        while (head != null) {
            // Check if the current node's value is in the set
            if (set.contains(head.val)) {
                // If it's the start of a new component, increment the count
                if (!inComponent) {
                    count++;
                    inComponent = true;
                }
            } else {
                // If we are no longer in a component, set inComponent to false
                inComponent = false;
            }

            // Move to the next node in the linked list
            head = head.next;
        }

        // Step 4: Return the count of connected components
        return count;
    }

    public static ListNode RemoveZeroSumConsecutiveNodesFromLinkedList(ListNode head) {
        // Create a dummy node that points to the head of the list
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        // Map to store the prefix sum and the corresponding node
        HashMap<Integer, ListNode> prefixSumMap = new HashMap<>();
        int prefixSum = 0;
        ListNode current = dummy;

        // Traverse the list and calculate the prefix sum
        while (current != null) {
            prefixSum += current.val;

            // If the prefix sum is already in the map, remove the intermediate nodes
            if (prefixSumMap.containsKey(prefixSum)) {
                ListNode node = prefixSumMap.get(prefixSum);
                // Remove all nodes between node and current (inclusive) from the list
                ListNode temp = node.next;
                int sum = prefixSum;
                while (temp != current) {
                    sum += temp.val;
                    prefixSumMap.remove(sum);
                    temp = temp.next;
                }
                node.next = current.next;
            } else {
                // Otherwise, add the prefix sum to the map
                prefixSumMap.put(prefixSum, current);
            }

            current = current.next;
        }

        // Return the list starting from the dummy's next node
        return dummy.next;
    }
    //endregion

    //region Private Methods

    private static ListNode getMiddle(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode slow = head;
        ListNode fast = head;

        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        return slow; //
    }

    private static ListNode merge(ListNode left, ListNode right) {
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while (left != null && right != null) {
            if (left.val <= right.val) {
                current.next = left;
                left = left.next;
            } else {
                current.next = right;
                right = right.next;
            }
            current = current.next;
        }

        // Attach the remaining nodes of either list
        if (left != null) {
            current.next = left;
        } else {
            current.next = right;
        }

        return dummy.next;
    }

    private static ListNode reverse(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    //endregion
}
