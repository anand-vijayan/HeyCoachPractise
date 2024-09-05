package org.mindmaps;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

public class Strings {
    //region Simple String
    public static int FinalValueOfVariableAfterPerformingOperations(String[] operations) {

        int x = 0;
        for (String operation : operations) {
            switch (operation) {
                case "--X":
                    --x;
                    break;
                case "X--":
                    x--;
                    break;
                case "++X":
                    ++x;
                    break;
                case "X++":
                    x++;
                    break;
                default:
                    break;
            }
        }

        return x;
    }

    public static String DefangingAnIPAddress(String address) {
        String output = "";
        for(char c : address.toCharArray()) {
            if(c == '.') {
                output += "[.]";
            } else {
                output += c;
            }
        }
        return output;
    }

    public static int MaximumNumberOfWordsFoundInSentences(String[] sentences) {
        int count = 0, maxCount = 0;
        for(String sentence : sentences) {
            count = 0;
            for(char c : sentence.toCharArray()) {
                if(c == ' ') count++;
            }
            //Since number of word is 1 more than number of white space
            count++;
            maxCount = Math.max(maxCount, count);
        }

        return maxCount;
    }

    public static String GoalParserInterpretation(String command) {
        StringBuilder result = new StringBuilder();
        int i = 0;

        while (i < command.length()) {
            if (command.charAt(i) == 'G') {
                result.append('G');
                i++;
            } else if (command.charAt(i) == '(' && command.charAt(i + 1) == ')') {
                result.append('o');
                i += 2;
            } else if (command.charAt(i) == '(' && command.charAt(i + 1) == 'a') {
                result.append("al");
                i += 4;
            }
        }

        return result.toString();
    }

    public static String ShuffleString(String s, int[] indices) {
        char[] result = new char[s.length()];

        // Place each character in its correct position
        for (int i = 0; i < s.length(); i++) {
            result[indices[i]] = s.charAt(i);
        }

        // Convert the character array to a string
        return new String(result);
    }
    //endregion

    //region Hashing
    //endregion

    //region Sorting
    //endregion

    //region Two Pointer
    //endregion

    //region Counting
    //endregion

    //region Sliding Window
    //endregion

    //region Bit Manipulation
    //endregion

    //region DP Based
    //endregion

    //region Prefix Sum
    //endregion

    //region Recursion
    //endregion

    //region Backtracking
    //endregion

    //region String Matching
    //endregion

    //region Private Methods
    //endregion
}
