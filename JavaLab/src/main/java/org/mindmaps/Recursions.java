package org.mindmaps;

import org.dto.ListNode;

public class Recursions {

    //region Recursion
    public static void ReverseString(char[] s) {
        Strings.ReverseString(s);
    }

    public static ListNode ReverseLinkedList(ListNode head) {
        return LinkedLists.ReverseLinkedList1(head);
    }

    public static int FibonacciNumber(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        // Initialize the first two Fibonacci numbers
        int a = 0, b = 1;

        // Iteratively compute the Fibonacci numbers up to n
        for (int i = 2; i <= n; i++) {
            int temp = a + b;
            a = b;
            b = temp;
        }

        return b;  // The nth Fibonacci number is stored in b
    }

    public static ListNode MergeTwoSortedLinkedList(ListNode list1, ListNode list2) {
        return LinkedLists.MergeTwoSortedLists(list1, list2);
    }

    public static boolean PalindromeLinkedList(ListNode head) {
        return Stacks.PalindromeLinkedList(head);
    }
    //endregion

}
