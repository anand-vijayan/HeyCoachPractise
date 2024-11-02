package org.heycoach;

import org.dto.ListNode;
import org.modules.AdvancedDataStructure;

import java.util.ArrayList;
import java.util.List;

import static org.sessiontests.Five.SpecialArrangements;
import static org.tgp.Level_C3.*;

public class Main {
    public static void main(String[] args)  {

        // Create a sample doubly linked list: 2 <-> 4 <-> 5
        ListNode head = new ListNode(1);
        ListNode node1 = new ListNode(2);
        //ListNode node2 = new ListNode(3);
        ListNode node2 = new ListNode(-1);
        head.next = node1;
        node1.prev = head;
        node1.next = node2;
        node2.prev = node1;
        //node2.next = node3;
        //node3.prev = node2;

        // Add a positive value to the list
        head = AdvancedDataStructure.AddNodeFirstMiddleAndLast(head, 9);

        // Print the modified list
        ListNode current = head;
        while (current != null) {
            System.out.print(current.val + " ");
            current = current.next;
        }
        // Output: 5 2 5 4 5 5
    }
}

