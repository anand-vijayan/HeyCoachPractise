package org.tgp;

import java.util.*;

public class Level_C1 {
    public static List<Integer> HelpTheGovernor(List<Integer> votes) {

        HashMap<Integer, Integer> voteCount = new HashMap<>();
        for(Integer i : votes) {
            voteCount.put(i, voteCount.getOrDefault(i, 0) + 1);
        }

        List<Integer> result = new ArrayList<>();

        for(Map.Entry<Integer,Integer> vote : voteCount.entrySet()) {
            if(vote.getValue() > 1) {
                result.add(vote.getKey());
            }
        }

        return result;
    }

    public static int RangeOfAnArray(int[] arr) {
        int maxValue = Integer.MIN_VALUE, minValue = Integer.MAX_VALUE;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] > maxValue) maxValue = arr[i];
            if(arr[i] < minValue) minValue = arr[i];
        }

        return maxValue - minValue;
    }

    public static String OnlyKString(String s, int k) {
        //Convert input string to character array and have a temp variable to hold string
        char[] originalTextAsChars = s.toCharArray();
        String temp = "";

        //Get the words in a list
        List<String> inputWords = new ArrayList<>();
        for(char c : originalTextAsChars) {
            if(c == ' ' && !temp.isEmpty()) {
                inputWords.add(temp);
                temp = "";
            } else{
                temp += c;
            }
        }
        temp = "";

        //Construct the final sentence with 'k' words
        int i = 0;
        while(i < k) {
            temp += inputWords.get(i) + " ";
            i++;
        }

        return temp.trim();
    }
}
