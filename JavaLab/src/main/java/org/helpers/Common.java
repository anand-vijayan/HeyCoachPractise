package org.helpers;

import org.dto.Node;

import java.util.ArrayList;
import java.util.List;

public class Common {
    public static void PrintArray(int[] arr) {
        StringBuilder output = new StringBuilder();
        for(int i = 0; i < arr.length; i++) {
            output.append(arr[i]).append(" ");
        }
        System.out.println(output.toString().trim());
    }

    public static void PrintArray(ArrayList<Integer> arr) {
        for(int i : arr) {
            System.out.print(i + " ");
        }
        System.out.println();
    }

    public static <T> void PrintArray(List<T> arr) {
        StringBuilder output = new StringBuilder();
        for(T i : arr) {
            output.append(i).append(" ");
        }
        System.out.println(output.toString().trim());
    }

    public static void PrintCircularList(Node head) {
        if (head == null) return;

        Node temp = head;
        do {
            System.out.print(temp.data + " -> ");
            temp = temp.next;
        } while (temp != head);
        System.out.println("(back to " + head.data + ")");
    }
}
