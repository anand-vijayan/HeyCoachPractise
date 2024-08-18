package org.helpers;

import java.util.*;

public class Sorting {
    public static int[] MergeSort(int[] inputArr) {

        return inputArr;
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
}
