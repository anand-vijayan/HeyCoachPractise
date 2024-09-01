package org.helpers;

import org.dto.BubbleSortData;

import java.util.*;

public class Sorting {
    public static void MergeSort(ArrayList<Integer> arr, int leftIndex, int middleIndex, int rightIndex) {

        int[] inputArr = arr.stream().mapToInt(i->i).toArray();

        //Calculate size of two temporary arrays which need to be merged and create them
        int sizeOfLeftArr = middleIndex - leftIndex + 1;
        int sizeOfRightArr = rightIndex - middleIndex;
        int[] leftArray = new int[sizeOfLeftArr];
        int[] rightArray = new int[sizeOfRightArr];

        //Fill data
        for(int i = 0; i < sizeOfLeftArr; i++) {
            leftArray[i] = inputArr[leftIndex + i];
        }
        for(int j = 0; j < sizeOfRightArr; j++) {
            rightArray[j] = inputArr[middleIndex + 1 + j];
        }

        //Merge the arrays with sorting (if required)
        int i = 0, j = 0, k = leftIndex;
        while(i < sizeOfLeftArr && j < sizeOfRightArr) {
            if(leftArray[i] <= rightArray[j]) {
                inputArr[k] = leftArray[i];
                i++;
            } else {
                inputArr[k] = rightArray[j];
                j++;
            }
            k++;
        }

        //Add elements which doesn't require sorting
        while(i < sizeOfLeftArr) {
            inputArr[k] = leftArray[i];
            i++;
            k++;
        }

        while(j < sizeOfRightArr) {
            inputArr[k] = rightArray[j];
            j++;
            k++;
        }

        arr.clear();
        for(int a : inputArr) {
            arr.add(a);
        }
    }

    public static int[] MergeSort(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length + arr2.length];
        int i = 0, j = 0, k = 0;

        while (i < arr1.length && j < arr2.length) {
            if(arr1[i] < arr2[j]){
                result[k] = arr1[i];
                i++;
                k++;
            } else if(arr1[i] > arr2[j]) {
                result[k] = arr2[j];
                j++;
                k++;
            } else {
                result[k] = arr1[i];
                k++;
                i++;
                result[k] = arr2[j];
                k++;
                j++;
            }
        }

        while(i < arr1.length) {
            result[k] = arr1[i];
            i++;
            k++;
        }

        while(j < arr2.length) {
            result[k] = arr2[j];
            j++;
            k++;
        }

        return result;
    }

    public static int[] BubbleSort(int[] arr) {
        boolean swapped = false;
        int n = arr.length;
        for(int i = 0; i < n - 1; i++) {
            swapped = false;
            for(int j = 0; j < n - i - 1; j++) {
                if(arr[j] > arr[j+1]) {
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                    swapped = true;
                }
            }
            if(!swapped) break;
        }
        return arr;
    }

    public static int[] BubbleSort(int[] arr, boolean invertSortToDescending) {
        boolean swapped = false;
        int n = arr.length;
        for(int i = 0; i < n - 1; i++) {
            swapped = false;
            for(int j = 0; j < n - i - 1; j++) {
                if(invertSortToDescending) {
                    if(arr[j] < arr[j+1]) {
                        int temp = arr[j];
                        arr[j] = arr[j+1];
                        arr[j+1] = temp;
                        swapped = true;
                    }
                } else {
                    if(arr[j] > arr[j+1]) {
                        int temp = arr[j];
                        arr[j] = arr[j+1];
                        arr[j+1] = temp;
                        swapped = true;
                    }
                }
            }
            if(!swapped) break;
        }
        return arr;
    }

    public static BubbleSortData BubbleSortWithNumberOfSwaps(int[] arr) {
        BubbleSortData output = new BubbleSortData();
        boolean swapped = false;
        int n = arr.length, swapCount = 0;
        for(int i = 0; i < n - 1; i++) {
            swapped = false;
            for(int j = 0; j < n - i - 1; j++) {
                if(arr[j] > arr[j+1]) {
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                    swapped = true;
                    swapCount++;
                }
            }
            if(!swapped) break;
        }
        output.SortedArray = arr;
        output.NumberOfSwaps = swapCount;
        return output;
    }

    public static void BucketSortString(String textToSort) {

        //1. Get the count
        Map<Character, Integer> dictionary = new HashMap<>();
        for(char c : textToSort.toCharArray()) {
            dictionary.put(c, dictionary.getOrDefault(c,0) + 1);
        }

        //2. Create buckets to hold the values
        int maxCount = Collections.max(dictionary.values());
        List<List<Character>> buckets = new ArrayList<>(maxCount + 1);
        for(int i = 0; i <= maxCount; i++) {
            buckets.add(new ArrayList<>());
        }

        //3. Fill the bucket with corresponding characters
        for(Map.Entry<Character,Integer> entry : dictionary.entrySet()) {
            char c = entry.getKey();
            int i = entry.getValue();
            buckets.get(i).add(c);
        }

        //4. Sort buckets in alphabetical order
        for(List<Character> bucket : buckets) {
            Collections.sort(bucket);
        }

        //5. Iterate the buckets to get final result
        StringBuilder sortedString = new StringBuilder();
        for (int i = maxCount; i >= 1; i--) {
            for (char c : buckets.get(i)) {
                for (int j = 0; j < i; j++) {
                    sortedString.append(c);
                }
            }
        }

        System.out.println(sortedString);
    }

    public static void BucketSortString(String textToSort, String pattern) {

        //1. Create a dictionary of characters in pattern, assuming no characters are repeated.
        Map<Character, Integer> patternDictionary = new HashMap<>();
        for(char c : pattern.toCharArray()) {
            patternDictionary.put(c,0);
        }

        //2. Get the count from textToSort
        for(char c : textToSort.toCharArray()) {
            patternDictionary.put(c, patternDictionary.getOrDefault(c,0) + 1);
        }

        //3. Print the characters in order of pattern and the frequency from input.
        StringBuilder output = new StringBuilder();
        for(Map.Entry<Character, Integer> entry : patternDictionary.entrySet()) {
            for(int i = 0; i < entry.getValue(); i++) {
                output.append(entry.getKey());
            }
        }

        System.out.println(output);
    }

    /**
     * Compare two words with case sensitivity.
     * @param word1 word1 without space
     * @param word2 word2 without space
     * @return 0=if both same, 1 or 2 = indicates the first word based on alphabetical order
     */
    public static int CompareWords(String word1, String word2) {
        char[] arr1 = word1.toCharArray();
        char[] arr2 = word2.toCharArray();
        int i = 0, j = 0;
        boolean sameWords = false;
        int firstWord = 0;

        while(i < arr1.length && j < arr2.length) {
            if(arr1[i] == arr2[j]) {
                i++;
                j++;
                sameWords = true;
            } else if(arr1[i] < arr2[j]) {
                firstWord = 1;
                sameWords = false;
                break;
            } else {
                firstWord = 2;
                sameWords = false;
                break;
            }
        }

        if(!sameWords) return firstWord;

        if(sameWords && i == arr1.length) {
            firstWord = 1;
        }

        if(sameWords && j == arr2.length) {
            firstWord = 2;
        }

        return firstWord;
    }
}
