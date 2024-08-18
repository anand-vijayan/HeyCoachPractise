package org.helpers;

import java.util.ArrayList;
import java.util.List;

public class Common {
    public static void PrintArray(int[] arr) {
        for(int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    public static void PrintArray(ArrayList<Integer> arr) {
        for(int i : arr) {
            System.out.print(i + " ");
        }
        System.out.println();
    }

    public static void PrintArray(List<String> arr) {
        for(String i : arr) {
            System.out.print(i + " ");
        }
        System.out.println();
    }
}
