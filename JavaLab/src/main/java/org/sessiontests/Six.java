package org.sessiontests;

import org.dto.Node;
import org.dto.TreeNode;

public class Six {
    public static Node ReverseALinkedList(Node head) {
        Node prev = null;
        Node next = null;

        while (head != null) {
            next = head.next;  // Save next node
            head.next = prev;  // Reverse the current node's pointer
            prev = head;       // Move prev and current one step forward
            head = next;
        }

        // prev will now be the new head of the reversed list
        return prev;
    }

    public static Node RotateList(Node head, int k) {
        if (head == null || head.next == null || k == 0) {
            return head;  // Edge cases: empty list, single node, or no rotation needed
        }

        // Step 1: Calculate the length of the list
        Node current = head;
        int length = 1;  // Start counting from head
        while (current.next != null) {
            current = current.next;
            length++;
        }

        // Step 2: Handle cases where k is larger than length
        k = k % length;
        if (k == 0) {
            return head;  // No rotation needed if k is a multiple of the list's length
        }

        // Step 3: Make the list circular by connecting the last node to the head
        current.next = head;

        // Step 4: Find the new tail (length - k steps from the start)
        int stepsToNewTail = length - k;
        Node newTail = head;
        for (int i = 1; i < stepsToNewTail; i++) {
            newTail = newTail.next;
        }

        // Step 5: The new head is the next node after the new tail
        Node newHead = newTail.next;
        newTail.next = null;  // Break the circular connection

        return newHead;  // Return the new head of the rotated list
    }

    public static int DiameterOfBinaryTree(TreeNode root){
        calculateHeight(root);
        return diameter;
    }

    //region Private Methods
    private static int diameter = 0;
    private static int calculateHeight(TreeNode root) {
        if (root == null) {
            return 0;  // Base case: the height of an empty subtree is 0
        }

        // Recursively calculate the height of the left and right subtrees
        int leftHeight = calculateHeight(root.left);
        int rightHeight = calculateHeight(root.right);

        // Update the diameter at this node
        diameter = Math.max(diameter, leftHeight + rightHeight);

        // Return the height of the current node's subtree
        return 1 + Math.max(leftHeight, rightHeight);
    }
    //endregion
}
