package org.sessiontests;

import java.util.ArrayList;
import java.util.List;

import static org.helpers.Sorting.*;

public class Two {
    public static int GetMedian(int[] ar1, int[] ar2, int n, int m) {
        int sum = 0;
        int[] mergedArray = MergeSort(ar1, ar2);
        int l = mergedArray.length;

        int[] middleIndexes = (l%2 == 0) ? new int[] {l/2, (l/2) - 1} : new int[] {l/2};

        for(int i : middleIndexes) {
            sum += mergedArray[i];
        }

        return middleIndexes.length > 1 ? sum/2 : sum;
    }

    public static int MangoesAndPineapples(String s, int n) {
        //Count the fruits
        int countOfMangoes = 0, countOfPineapples = 0;
        for(char c: s.toCharArray()) {
            if(c == 'M') countOfMangoes++;
            else if(c == 'P') countOfPineapples++;
        }

        //Iterate through array and count prefix mangoes and pineapples
        //Match it with total mangoes and pineapples, if any left apply rest of the condition
        int prefixMangoes = 0, prefixPineapples = 0;
        for(int i = 0; i < n - 1; i++) {
            if(s.charAt(i) == 'M') prefixMangoes++;
            else if(s.charAt(i) == 'P') prefixPineapples++;

            int remainingMangoes = countOfMangoes - prefixMangoes;
            int remainingPineapples = countOfPineapples - prefixPineapples;

            //Check final condition
            if(prefixMangoes != remainingMangoes && prefixPineapples != remainingPineapples) {
                return i + 1;
            }
        }

        return -1;
    }

    public static void SwapToMax(int n, int[] a, int[] b) {
        for (int i = 0; i < n; i++) {
            // Ensure that both arrays have their maximums placed towards the end
            if (a[i] > a[n-1] || b[i] > b[n-1]) {
                // Swap to make sure the maximum values can be placed at the end
                int temp = a[i];
                a[i] = b[i];
                b[i] = temp;
            }
        }

        // Now, check if the last elements are the maximum in their respective arrays
        int maxA = a[0], maxB = b[0];
        for (int i = 1; i < n; i++) {
            if (a[i] > maxA) maxA = a[i];
            if (b[i] > maxB) maxB = b[i];
        }

        if (a[n-1] == maxA && b[n-1] == maxB) {
            System.out.println("YES");
        } else {
            System.out.println("NO");
        }
    }

}
