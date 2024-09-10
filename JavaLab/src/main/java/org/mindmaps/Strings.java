package org.mindmaps;

import org.modules.BasicDataStructures;

import java.util.*;

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
    public static int JewelsAndStones(String jewels, String stones) {
        TreeSet<Character> jewelCount = new TreeSet<>();
        int count = 0;

        //Enter the valid jewels
        for(char jewel : jewels.toCharArray()) {
            jewelCount.add(jewel);
        }

        //Increment the count if those jewel found in second string
        for(char stone : stones.toCharArray()) {
            if(jewelCount.contains(stone)) {
                count++;
            }
        }

        return count;
    }

    public int CountTheNumberOfConsistentStrings(String allowed, String[] words) {
        return Array.CountTheNumberOfConsistentStrings(allowed, words);
    }

    public static int RingsAndRods(String rings) {
        int n = rings.length();

        //String length must be 6 or more
        if(n < 6) {
            return 0;
        }

        //Have an array of strings to arrange the rings
        String[] rods = new String[10];
        Set<Integer> completedRods = new HashSet<>();

        //Arrange or fill the array based on integer value next to the character
        for(int i = 0; i < n - 1; i += 2){
            String indexChar = String.valueOf(rings.charAt(i + 1));
            int index = Integer.parseInt(indexChar);
            if(index > 9) return 0;

            if(rods[index] == null) {
                rods[index] = String.valueOf(rings.charAt(i));
            } else {
                rods[index] += rings.charAt(i);
            }

            if(rods[index].contains("R")
                    && rods[index].contains("G")
                    && rods[index].contains("B")) {
                completedRods.add(index);
            }
        }

        return completedRods.size();
    }

    public static boolean CheckIfTheSentenceIsPangram(String sentence) {
        if(sentence.length() < 26) return false;

        Set<Character> availableChars = new HashSet<>();
        for(char c : sentence.toCharArray()) {
            availableChars.add(Character.toLowerCase(c));
        }

        return availableChars.size() == 26;
    }

    public static String IncreasingDecreasingString_1(String s) {
        int[] count = new int[26];  // Array to store character frequencies.

        // Count frequency of each character.
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }

        StringBuilder result = new StringBuilder();
        int length = s.length();

        while (result.length() < length) {
            // Step 1 & Step 2: Add smallest to largest characters
            for (int i = 0; i < 26; i++) {
                if (count[i] > 0) {
                    result.append((char)(i + 'a'));
                    count[i]--;
                }
            }
            // Step 4 & Step 5: Add largest to smallest characters
            for (int i = 25; i >= 0; i--) {
                if (count[i] > 0) {
                    result.append((char)(i + 'a'));
                    count[i]--;
                }
            }
        }

        return result.toString();
    }
    //endregion

    //region Sorting
    public static String SortingTheSentence(String s) {
        String[] words = s.split(" ");
        String[] sortedWords = new String[words.length];
        for (String word : words) {
            int position = word.charAt(word.length() - 1) - '1';
            sortedWords[position] = word.substring(0, word.length() - 1);
        }
        return String.join(" ", sortedWords);
    }

    public static boolean ValidAnagram(String s, String t) {
        //Count the characters in both words, if they are same then anagram else false
        int[] sCount = new int[26];
        int[] tCount = new int[26];

        //Count the frequency
        for(char c : s.toLowerCase().toCharArray()) {
            sCount[c - 'a']++;
        }

        for(char c : t.toLowerCase().toCharArray()) {
            tCount[c - 'a']++;
        }

        //Now match both
        for(int i = 0; i < 26; i++) {
            if(sCount[i] != tCount[i]) return false;
        }
        return true;
    }

    public static char FindTheDifference(String s, String t) {

        //Original solution
        //Count the characters in both words, if they are same then anagram else false
        int[] sCount = new int[26];
        int[] tCount = new int[26];

        //Count the frequency
        for(char c : s.toLowerCase().toCharArray()) {
            sCount[c - 'a']++;
        }

        for(char c : t.toLowerCase().toCharArray()) {
            tCount[c - 'a']++;
        }

        //Now match both
        for(int i = 0; i < 26; i++) {
            if(sCount[i] != tCount[i]) return (char) (i + 'a');
        }
        return ' ';
        //Hack (this won't work if character or word repeats
        //return t.replace(s,"").toCharArray()[0];
    }

    public static List<String> FindAndReplacePattern(String[] words, String pattern){
        List<String> matchingWords = new ArrayList<>();
        for(String word : words){
            if(FindAndReplacePatternHelper(word,pattern)){
                matchingWords.add(word);
            }
        }
        return matchingWords;
    }

    public static String SortCharactersByFrequency(String s) {
        //1. Count the frequency of each character
        Map<Character, Integer> frequencyMap = new HashMap<>();
        for (char c : s.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }

        //2. Sort characters by frequency
        List<Character> characters = new ArrayList<>(frequencyMap.keySet());
        characters.sort((a, b) -> frequencyMap.get(b) - frequencyMap.get(a));

        //3. Build the result string
        StringBuilder result = new StringBuilder();
        for (char c : characters) {
            int freq = frequencyMap.get(c);
            for (int i = 0; i < freq; i++) {
                result.append(c);
            }
        }

        return result.toString();
    }
    //endregion

    //region Two Pointer
    public static String FindFirstPalindromicStringInTheArray(String[] words) {
        return Array.FindFirstPalindromicStringInTheArray(words);
    }

    public static String ReverseWordsInAString_3(String s) {
        StringBuilder output = new StringBuilder();
        String[] words = s.split(" ");
        for (String word : words) {
            output.append(ReverseAWord(word.trim())).append(" ");
        }
        return output.toString().trim();
    }

    public static String ReversePrefixOfWord(String word, char c) {
        //If character not found return same word
        if(!word.contains(String.valueOf(c))) return word;

        //Create a stack of characters till 'c'
        Stack<Character> wordStack = new Stack<>();
        int positionOfCharC = 0;
        for(char a : word.toCharArray()) {
            wordStack.push(a);
            positionOfCharC++;
            if(a == c) {
                break;
            }
        }

        //Make the output
        StringBuilder output = new StringBuilder();
        while (!wordStack.empty()) {
            output.append(wordStack.pop());
        }
        output.append(word, positionOfCharC, word.length());
        return output.toString();
    }

    public static int[] DIStringMatch(String s){
        return Array.DIStringMatch(s);
    }

    public static String MergeStringsAlternately(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();

        if(m > 0 && n > 0) {
            char[] charactersOfWord1 = word1.toCharArray();
            char[] charactersOfWord2 = word2.toCharArray();
            int limit = Math.max(m,n);
            StringBuilder output = new StringBuilder();
            for(int i = 0; i < limit; i++) {
                if(i < m) {
                    output.append(charactersOfWord1[i]);
                }

                if(i < n) {
                    output.append(charactersOfWord2[i]);
                }
            }
            return output.toString();
        } else {
            return "";
        }
    }
    //endregion

    //region Counting
    public static int SplitAStringInBalancedStrings(String s){
        int countOfL = 0, countOfR = 0, balanceStringCount = 0;
        for(char c : s.toCharArray()) {
            if(c == 'L') countOfL++;
            if(c == 'R') countOfR++;
            if(countOfR == countOfL) balanceStringCount++;
        }
        return balanceStringCount;
    }

    public static boolean DetermineIfStringHalvesAreAlike(String s){
        int n = s.length(), m = n/2, count1 = 0, count2 = 0;
        for (int i = 0; i < n; i++) {
            if(i < m && BasicDataStructures.IsVowel(s.charAt(i))) count1++;
            else if(i >= m && BasicDataStructures.IsVowel(s.charAt(i))) count2++;
            else continue;
        }
        return count1 == count2;
    }

    public static String IncreasingDecreasingString_2(String s) {
        return IncreasingDecreasingString_1(s);
    }

    public static boolean CheckIfAllCharactersHaveEqualNumberOfOccurrences(String s){
        //TreeMap to hold frequency
        TreeMap<Character, Integer> frequency = new TreeMap<>();
        for(char c : s.toCharArray()) {
            frequency.put(c, frequency.getOrDefault(c, 0) + 1);
        }

        //Traverse frequency to check if there is more than one count
        int currentCount = Collections.max(frequency.values());
        for(Map.Entry<Character,Integer> f : frequency.entrySet()) {
            if(currentCount != f.getValue()) return false;
        }
        return true;
    }

    public static String KthDistinctStringInAnArray(String[] arr, int k){
        // Step 1: Count the frequency of each string
        HashMap<String, Integer> frequencyMap = new HashMap<>();

        for (String s : arr) {
            frequencyMap.put(s, frequencyMap.getOrDefault(s, 0) + 1);
        }

        // Step 2: Filter distinct strings and find the k-th one
        int distinctCount = 0;
        for (String s : arr) {
            if (frequencyMap.get(s) == 1) {
                distinctCount++;
                if (distinctCount == k) {
                    return s;  // Return the k-th distinct string
                }
            }
        }

        return "";  // If there are fewer than k distinct strings, return an empty string
    }

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
    private static boolean FindAndReplacePatternHelper(String word, String pattern) {
        if (word.length() != pattern.length()) {
            return false;
        }

        // Map for word -> pattern mapping
        Map<Character, Character> wordToPattern = new HashMap<>();
        // Map for pattern -> word mapping
        Map<Character, Character> patternToWord = new HashMap<>();

        for (int i = 0; i < word.length(); i++) {
            char w = word.charAt(i);
            char p = pattern.charAt(i);

            // Check if there's already a mapping in wordToPattern
            if (wordToPattern.containsKey(w)) {
                // If the current mapping does not match the pattern, return false
                if (wordToPattern.get(w) != p) {
                    return false;
                }
            } else {
                wordToPattern.put(w, p);
            }

            // Check if there's already a mapping in patternToWord
            if (patternToWord.containsKey(p)) {
                // If the current mapping does not match the word, return false
                if (patternToWord.get(p) != w) {
                    return false;
                }
            } else {
                patternToWord.put(p, w);
            }
        }

        return true;
    }

    private static String ReverseAWord(String s) {
        //This method is to reverse a word, so if there is white space then return same string
        if(s.contains(" ")) return s;

        //Use stack to hold characters
        Stack<Character> wordStack = new Stack<>();
        for(char c : s.toCharArray()){
            wordStack.push(c);
        }

        //Now, form output from stack
        StringBuilder output = new StringBuilder();
        while (!wordStack.empty()){
            output.append(wordStack.pop());
        }

        return output.toString();
    }
    //endregion
}
