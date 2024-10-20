package org.modules;

import org.dto.ListNode;
import org.dto.Node;

import java.util.List;
import java.util.Stack;

public class AdvancedDataStructure {
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
    //endregion
}
