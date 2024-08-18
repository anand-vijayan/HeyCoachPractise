package org.heycoach;

import static org.helpers.Common.PrintArray;
import static org.helpers.Sorting.BubbleSort;

public class Samples {

    public static void MaximumStonesGame(int[] inputArr, int maxNumOfPiles) {
        int maxStones = 0, n = inputArr.length;
        inputArr = BubbleSort(inputArr);
        for (int i = n / maxNumOfPiles; i < n; i += maxNumOfPiles-1) {
            maxStones += inputArr[i];
        }

        System.out.println(maxStones);
    }

    public static void MakeWave(int[] inputArr) {
        inputArr = BubbleSort(inputArr);
        for(int i = 0; i < inputArr.length - 1; i+=2) {
            int temp = inputArr[i+1];
            inputArr[i+1] = inputArr[i];
            inputArr[i] = temp;
        }

        PrintArray(inputArr);
    }

}
