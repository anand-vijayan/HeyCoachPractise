package org.heycoach;

import java.util.Vector;

import static org.sessiontests.Seven.GoldNuggets;

public class Main {
    public static void main(String[] args)  {
        int[] arr = new int[] {3,1,6,6,3,6};
        Vector<Integer> inputVector = new Vector<>();
        for(int a : arr) {
            inputVector.add(a);
        }
        System.out.println(GoldNuggets(inputVector));
    }
}

