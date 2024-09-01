package org.helpers;

import org.dto.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

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

    public static <T> void PrintArray(Queue<T> queue) {
        StringBuilder output = new StringBuilder();
        while(!queue.isEmpty()) {
            output.append(queue.poll()).append(" ");
        }
        System.out.println(output.toString().trim());
    }

    public static void PrintArray(int[][] arr) {
        StringBuilder output = new StringBuilder();
        output.append("[");
        for(int i = 0; i < arr.length; i++) {
            output.append((i > 0) ? ",[" : "[");
            for(int j = 0; j < arr[i].length; j++) {
                output.append(arr[i][j]).append((j + 1 == arr[i].length) ? "" : ",");
            }
            output.append("]");
        }
        output.append("]");
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
