package org.tgp;

import java.util.*;

public class Level_C3 {
    public static int BasicCalculator(String s) {
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        int number = 0;
        int sign = 1; // 1 for positive, -1 for negative

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);

            // If the character is a digit, construct the number
            if (Character.isDigit(ch)) {
                number = number * 10 + (ch - '0');
            }
            // If we encounter a + or -, add the previous number to the result and update the sign
            else if (ch == '+') {
                result += sign * number;
                number = 0; // Reset number
                sign = 1;   // Update sign for next number
            }
            else if (ch == '-') {
                result += sign * number;
                number = 0; // Reset number
                sign = -1;  // Update sign for next number
            }
            // If we encounter an opening parenthesis, push the result and sign to the stack
            else if (ch == '(') {
                stack.push(result); // Push the result before '('
                stack.push(sign);   // Push the sign before '('
                result = 0;         // Reset result for the expression inside parentheses
                sign = 1;           // Reset sign for inside the parentheses
            }
            // If we encounter a closing parenthesis, calculate the result inside the parentheses
            else if (ch == ')') {
                result += sign * number;   // Add the current number to the result
                number = 0;                // Reset number

                result *= stack.pop();     // Multiply the result inside the parentheses by the sign before '('
                result += stack.pop();     // Add the result before the parentheses
            }
        }

        // There may be a remaining number to be added after loop ends
        if (number != 0) {
            result += sign * number;
        }

        return result;
    }

    public static String SortTheArray(String s){
        // Step 1: Count frequency of each character
        Map<Character, Integer> frequencyMap = new HashMap<>();
        for (char c : s.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }

        // Step 2: Convert map entries to a list and sort by frequency (descending), and then by character (ascending)
        List<Map.Entry<Character, Integer>> entries = new ArrayList<>(frequencyMap.entrySet());

        // Custom sorting: first by frequency in descending order, then by character in ascending order
        entries.sort((a, b) -> {
            if (!a.getValue().equals(b.getValue())) {
                return b.getValue() - a.getValue();  // Sort by frequency in descending order
            } else {
                return a.getKey() - b.getKey();  // Sort alphabetically if frequencies are the same
            }
        });

        // Step 3: Build the resulting string based on sorted entries
        StringBuilder result = new StringBuilder();
        for (Map.Entry<Character, Integer> entry : entries) {
            char c = entry.getKey();
            int frequency = entry.getValue();
            for (int i = 0; i < frequency; i++) {
                result.append(c);  // Append character 'frequency' number of times
            }
        }

        return result.toString();
    }

    public static List<List<String>> GroupAnangrams(String[] strs) {
        // Step 1: Create a map to store sorted word as key and its anagrams as values
        Map<String, List<String>> anagramMap = new HashMap<>();

        for (String str : strs) {
            // Step 2: Sort the string to use as key
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String sortedStr = new String(chars);

            // Step 3: Add the original string to the list of its sorted key
            anagramMap.computeIfAbsent(sortedStr, k -> new ArrayList<>()).add(str);
        }

        // Step 4: Collect all anagram groups and sort them lexicographically
        List<List<String>> groupedAnagrams = new ArrayList<>(anagramMap.values());

        for (List<String> group : groupedAnagrams) {
            // Sort each group lexicographically
            Collections.sort(group);
        }

        // Step 5: Sort the entire list of groups lexicographically based on the first element of each group
        groupedAnagrams.sort((a, b) -> a.get(0).compareTo(b.get(0)));

        // Step 6: Print each group with a space between the words
        for (List<String> group : groupedAnagrams) {
            System.out.println(String.join(" ", group));
        }

        return groupedAnagrams;
    }
}
