package org.modules;

import org.dto.ListNode;
import org.dto.Node;
import org.helpers.Common;

import java.util.ArrayList;
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
        return AddNode(head, val);
    }
    //endregion

    //region Binary Trees
    //endregion

    //region Private Methods
    private static ListNode AddNode(ListNode head, int val) {
        //Step 1: Read the values into a list, if any value is negative, then ignore it.
        List<Integer> basicValues = new ArrayList<>();
        ListNode current = head;

        while (current != null) {
            if (current.val > 0) {
                basicValues.add(current.val);
            }
            // Move to the next node
            current = current.next;
        }

        //Step 2: Replicate the final node in an array and Insert values at front, middle and last.
        int[] valuesArray = new int[basicValues.size() + 3];
        int middleIndex = basicValues.size()/2 + 1;
        int j = 0;

        for(int i = 0; i < valuesArray.length; i++) {
            if(i == 0 || i == middleIndex || i == valuesArray.length - 1) {
                valuesArray[i] = val;
            } else {
                valuesArray[i] = basicValues.get(j);
                j++;
            }
        }

        Common.PrintArray(valuesArray);

        //Step 3: Create final list node
        head = new ListNode(valuesArray[0]);
        current = head;

        for (int i = 1; i < valuesArray.length; i++) {
            current.next = new ListNode(valuesArray[i]); // Create a new node
            current = current.next; // Move to the next node
        }

        return head; // Return the modified list
    }
    //endregion
}
