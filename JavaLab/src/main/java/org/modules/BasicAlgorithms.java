/* ****************************************************************************************************** **
*  @@@@@@     Building blocks and basic algorithms     @@@@@@
*  Explore the foundational elements of computer science, including essential algorithms and
*  sorting techniques that are crucial for tech interviews.
*  ****************************************************************************************************** */

package org.modules;

import org.dto.BubbleSortData;

import java.util.ArrayList;
import java.util.List;

import static org.helpers.Common.PrintArray;
import static org.helpers.Sorting.*;

public class BasicAlgorithms {

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

    public static int BubbleSortStoneGame(int[] player1, int[] player2){

        BubbleSortData player1Result = BubbleSortWithNumberOfSwaps(player1);
        BubbleSortData player2Result = BubbleSortWithNumberOfSwaps(player2);

        if(player1Result.NumberOfSwaps < player2Result.NumberOfSwaps) return 1;
        if(player2Result.NumberOfSwaps < player1Result.NumberOfSwaps) return 2;

        return 0;
    }

    public static void SortUsingMerge(ArrayList<Integer> inputArr, int leftIndex, int rightIndex){
        if(leftIndex < rightIndex) {
            //Get middle
            int middleIndex = leftIndex + (rightIndex - leftIndex) / 2;

            //Sort Left and right half
            SortUsingMerge(inputArr, leftIndex, middleIndex);
            SortUsingMerge(inputArr, middleIndex + 1, rightIndex);

            //Merge the sorted arrays
            MergeSort(inputArr, leftIndex, middleIndex, rightIndex);
        }
    }

    public static List<String> CombineTheBookLists(List<String> sortedInventory1, List<String> sortedInventory2) {
        List<String> combinedInventory = new ArrayList<>();
        String[] inventory1 = sortedInventory1.toArray(new String[0]);
        String[] inventory2 = sortedInventory2.toArray(new String[0]);
        int firstWord, i = 0, j = 0, k = 0;

        while(i < sortedInventory1.size() && j < sortedInventory2.size()) {
            firstWord = CompareWords(inventory1[i], inventory2[j]);
            switch (firstWord) {
                case 0:
                    combinedInventory.add(inventory1[i]); i++;
                    combinedInventory.add(inventory2[j]); j++;
                    break;
                case 1:
                    combinedInventory.add(inventory1[i]); i++;
                    break;
                case 2:
                    combinedInventory.add(inventory2[j]); j++;
                    break;
            }
        }

        while(i < sortedInventory1.size()) {
            combinedInventory.add(inventory1[i]);
            i++;
        }

        while(j < sortedInventory2.size()) {
            combinedInventory.add(inventory2[j]);
            j++;
        }

        return combinedInventory;
    }
}
