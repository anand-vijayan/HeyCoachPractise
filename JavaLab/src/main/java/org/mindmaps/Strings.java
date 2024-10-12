package org.mindmaps;

import org.dto.TreeNode;
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
        StringBuilder output = new StringBuilder();
        for(char c : address.toCharArray()) {
            if(c == '.') {
                output.append("[.]");
            } else {
                output.append(c);
            }
        }
        return output.toString();
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

    public int CountTheNumberOfConsistentStrings_1(String allowed, String[] words) {
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

    public static char FindTheDifference_1(String s, String t) {

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
            result.append(String.valueOf(c).repeat(Math.max(0, freq)));
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
    public static int SubstringsOfSizeThreeWithDistinctCharacters(String s) {
        int n = s.length();
        int windowSize = 3;
        int validSubString = 0;
        int m = n-windowSize+1;
        TreeMap<Character,Integer> count = new TreeMap<>();
        for(int i = 0; i < m; i++) {
            String word = s.substring(i, i + windowSize);
            System.out.println(word);
            for(char c : word.toCharArray()){
                count.put(c, count.getOrDefault(c, 0) + 1);
            }
            if(Collections.max(count.values()) == 1) validSubString++;
            count.clear();
        }

        return validSubString;
    }

    public static String LongestNiceSubstring_1(String s) {
        return LongestNiceSubstringHelper(s, 0, s.length());
    }

    public static int NumberOfSubstringsContainingAllThreeCharacters(String s) {
        int n = s.length();
        int[] count = new int[3]; // Array to store the count of 'a', 'b', 'c'
        int left = 0;
        int result = 0;

        for (int right = 0; right < n; right++) {
            // Increment the count for the current character
            count[s.charAt(right) - 'a']++;

            // While the window contains at least one of each character 'a', 'b', and 'c'
            while (count[0] > 0 && count[1] > 0 && count[2] > 0) {
                // All substrings starting from 'left' to 'right' are valid
                result += n - right; // (n - right) gives the number of valid substrings

                // Shrink the window from the left
                count[s.charAt(left) - 'a']--;
                left++;
            }
        }

        return result;
    }

    public static int MaximumNumberOfVowelsInASubstringOfGivenLength(String s, int k) {
        // A set containing all vowel characters
        Set<Character> vowels = Set.of('a', 'e', 'i', 'o', 'u');
        int maxVowels = 0; // To store the maximum number of vowels
        int currentVowels = 0; // To store the number of vowels in the current window

        // Initialize the first window of size k
        for (int i = 0; i < k; i++) {
            if (vowels.contains(s.charAt(i))) {
                currentVowels++;
            }
        }
        maxVowels = currentVowels;

        // Slide the window across the string
        for (int i = k; i < s.length(); i++) {
            // Remove the leftmost character from the window
            if (vowels.contains(s.charAt(i - k))) {
                currentVowels--;
            }

            // Add the new character to the window
            if (vowels.contains(s.charAt(i))) {
                currentVowels++;
            }

            // Update the maximum number of vowels encountered so far
            maxVowels = Math.max(maxVowels, currentVowels);
        }

        return maxVowels;
    }

    public static int MaximizeTheConfusionOfAnExam_1(String answerKey, int k){
        return Math.max(MaximizeTheConfusionOfAnExamHelper(answerKey,k,'T'), MaximizeTheConfusionOfAnExamHelper(answerKey,k,'F'));
    }
    //endregion

    //region Bit Manipulation
    public static int CountTheNumberOfConsistentStrings_2(String allowed, String[] words) {
        // Convert allowed characters into a Set for quick lookup
        Set<Character> allowedSet = new HashSet<>();
        for (char c : allowed.toCharArray()) {
            allowedSet.add(c);
        }

        int consistentCount = 0; // To count the number of consistent strings

        // Iterate through each word in the words array
        for (String word : words) {
            boolean isConsistent = true;

            // Check if all characters in the word are in the allowed set
            for (char c : word.toCharArray()) {
                if (!allowedSet.contains(c)) {
                    isConsistent = false;
                    break; // If we find a character not in the allowed set, the word is not consistent
                }
            }

            // If the word is consistent, increment the count
            if (isConsistent) {
                consistentCount++;
            }
        }

        return consistentCount;
    }

    public static String LongestNiceSubstring_2(String s){
        return LongestNiceSubstring_1(s);
    }

    public static char FindTheDifference_2(String s, String t){
        char result = 0;

        // XOR all characters in s
        for (char c : s.toCharArray()) {
            result ^= c;
        }

        // XOR all characters in t
        for (char c : t.toCharArray()) {
            result ^= c;
        }

        // The result will be the extra character in t
        return result;
    }

    public static String AddBinary(String a, String b){
        StringBuilder result = new StringBuilder();
        int i = a.length() - 1; // Pointer for string a
        int j = b.length() - 1; // Pointer for string b
        int carry = 0; // To store the carry during addition

        // Process both strings from the end
        while (i >= 0 || j >= 0) {
            int sum = carry; // Start with the carry

            // Add current digit of a, if available
            if (i >= 0) {
                sum += a.charAt(i) - '0';
                i--;
            }

            // Add current digit of b, if available
            if (j >= 0) {
                sum += b.charAt(j) - '0';
                j--;
            }

            // Append the current bit to the result (0 or 1)
            result.append(sum % 2);

            // Update carry (sum divided by 2)
            carry = sum / 2;
        }

        // If there's still a carry, append it
        if (carry != 0) {
            result.append(carry);
        }

        // Reverse the result since we've built it backwards
        return result.reverse().toString();
    }

    public static List<String> LetterCasePermutation_1(String s){
        List<String> result = new ArrayList<>();
        LetterCasePermutationHelper(s.toCharArray(), 0, result);
        return result;
    }
    //endregion

    //region DP Based
    public static boolean IsSubsequence(String s, String t){
        int sLen = s.length(), tLen = t.length();
        int sPointer = 0, tPointer = 0;

        // Traverse the string t
        while (sPointer < sLen && tPointer < tLen) {
            // If characters match, move both pointers
            if (s.charAt(sPointer) == t.charAt(tPointer)) {
                sPointer++;
            }
            // Always move the pointer of t
            tPointer++;
        }

        // If we have traversed all characters of s, return true
        return sPointer == sLen;
    }

    public static int CountSubstringsThatDifferByOneCharacter(String s, String t) {
        int sLen = s.length();
        int tLen = t.length();
        int count = 0;

        // Compare each substring from s with each substring from t
        for (int i = 0; i < sLen; i++) {
            for (int j = 0; j < tLen; j++) {
                int mismatchCount = 0;

                // Compare substrings of different lengths starting at i and j
                for (int k = 0; i + k < sLen && j + k < tLen; k++) {
                    if (s.charAt(i + k) != t.charAt(j + k)) {
                        mismatchCount++;
                    }

                    // If exactly one mismatch is found, count it
                    if (mismatchCount == 1) {
                        count++;
                    } else if (mismatchCount > 1) {
                        break; // Stop early if more than one mismatch is found
                    }
                }
            }
        }

        return count;
    }

    public static List<String> GenerateParentheses(int n) {
        List<String> result = new ArrayList<>();
        GenerateParenthesesHelper(result, "", 0, 0, n);
        return result;
    }

    public static int NumberOfGoodWaysToSplitAString(String s) {
        int n = s.length();
        if (n < 2) return 0; // No valid splits if the length of the string is less than 2

        // Arrays to store the number of distinct characters in prefix and suffix
        int[] prefixDistinctCount = new int[n];
        int[] suffixDistinctCount = new int[n];

        // Compute distinct characters count for prefixes
        Set<Character> prefixSet = new HashSet<>();
        for (int i = 0; i < n; i++) {
            prefixSet.add(s.charAt(i));
            prefixDistinctCount[i] = prefixSet.size();
        }

        // Compute distinct characters count for suffixes
        Set<Character> suffixSet = new HashSet<>();
        for (int i = n - 1; i >= 0; i--) {
            suffixSet.add(s.charAt(i));
            suffixDistinctCount[i] = suffixSet.size();
        }

        // Count good splits
        int goodSplits = 0;
        for (int i = 0; i < n - 1; i++) {
            if (prefixDistinctCount[i] == suffixDistinctCount[i + 1]) {
                goodSplits++;
            }
        }

        return goodSplits;
    }

    public static List<Integer> DifferentWaysToAddParentheses_1(String expression){
        if (memo.containsKey(expression)) {
            return memo.get(expression);
        }

        List<Integer> result = new ArrayList<>();

        // Base case: if the expression is just a number, return it as a single result
        if (!expression.contains("+") && !expression.contains("-") && !expression.contains("*")) {
            result.add(Integer.parseInt(expression));
            memo.put(expression, result);
            return result;
        }

        // Recursive case: split the expression by operators and compute results
        for (int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                // Divide the expression into left and right parts based on the current operator
                String leftExpr = expression.substring(0, i);
                String rightExpr = expression.substring(i + 1);

                // Recursively compute results for the left and right parts
                List<Integer> leftResults = DifferentWaysToAddParentheses_1(leftExpr);
                List<Integer> rightResults = DifferentWaysToAddParentheses_1(rightExpr);

                // Combine results from left and right parts based on the current operator
                for (int left : leftResults) {
                    for (int right : rightResults) {
                        if (c == '+') {
                            result.add(left + right);
                        } else if (c == '-') {
                            result.add(left - right);
                        } else {
                            result.add(left * right);
                        }
                    }
                }
            }
        }

        memo.put(expression, result);
        return result;
    }
    //endregion

    //region Prefix Sum
    public static int FindTheLongestSubstringContainingVowelsInEvenCounts(String s){
        // Map to store the first occurrence index of each bitmask
        Map<Integer, Integer> prefixMap = new HashMap<>();
        prefixMap.put(0, -1); // Base case: empty substring

        int maxLength = 0;
        int bitmask = 0; // Initial bitmask (all vowels counts are even)

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // Update bitmask if the character is a vowel
            switch (c) {
                case 'a':
                    bitmask ^= 1; // Toggle bit 0
                    break;
                case 'e':
                    bitmask ^= 1 << 1; // Toggle bit 1
                    break;
                case 'i':
                    bitmask ^= 1 << 2; // Toggle bit 2
                    break;
                case 'o':
                    bitmask ^= 1 << 3; // Toggle bit 3
                    break;
                case 'u':
                    bitmask ^= 1 << 4; // Toggle bit 4
                    break;
            }

            // Check if the current bitmask has been seen before
            if (prefixMap.containsKey(bitmask)) {
                int prevIndex = prefixMap.get(bitmask);
                maxLength = Math.max(maxLength, i - prevIndex);
            } else {
                // Store the first occurrence of this bitmask
                prefixMap.put(bitmask, i);
            }

            // Check all possible single bit flips (changing exactly one vowel's parity)
            for (int j = 0; j < 5; j++) {
                int flippedMask = bitmask ^ (1 << j);
                if (prefixMap.containsKey(flippedMask)) {
                    int prevIndex = prefixMap.get(flippedMask);
                    maxLength = Math.max(maxLength, i - prevIndex);
                }
            }
        }

        return maxLength;
    }

    public static int MaximizeTheConfusionOfAnExam_2(String answerKey, int k){
        return MaximizeTheConfusionOfAnExam_1(answerKey,k);
    }

    public static int MinimumWhiteTilesAfterCoveringWithCarpets(String s, int nc, int l){
        int n = s.length();
        int[][] dp = new int[n + 1][nc + 1];
        for (int i = 1; i <= n; ++i) {
            for (int k = 0; k <= nc; ++k) {
                int jump = dp[i - 1][k] + s.charAt(i - 1) - '0';
                int cover = k > 0 ? dp[Math.max(i - l, 0)][k - 1] : 1000;
                dp[i][k] = Math.min(cover, jump);
            }
        }
        return dp[n][nc];
    }
    //endregion

    //region Recursion
    public static void ReverseString(char[] s){
        int left = 0;
        int right = s.length - 1;

        while (left < right) {
            // Swap characters at left and right indices
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;

            // Move the pointers towards the center
            left++;
            right--;
        }
    }

    public static List<Integer> DifferentWaysToAddParentheses_2(String expression){
        return DifferentWaysToAddParentheses_1(expression);
    }

    public static List<List<String>> PalindromePartitioning(String s) {
        List<List<String>> result = new ArrayList<>();
        List<String> currentPartition = new ArrayList<>();
        PalindromePartitioningHelper(s, 0, currentPartition, result);
        return result;
    }

    public static char FindKthBitInNthBinaryString(int n, int k){
        return FindKthBitInNthBinaryStringHelper(n,k);
    }

    public static int MaximumScoreWordsFormedByLetters(String[] words, char[] letters, int[] score) {
        // Count the frequency of each letter in the list of letters
        Map<Character, Integer> letterCount = new HashMap<>();
        for (char c : letters) {
            letterCount.put(c, letterCount.getOrDefault(c, 0) + 1);
        }

        // List to store valid words and their scores
        List<String> validWords = new ArrayList<>();
        Map<String, Integer> wordScores = new HashMap<>();

        for (String word : words) {
            if (CanFormWord(word, letterCount)) {
                int wordScore = GetWordScore(word, score);
                validWords.add(word);
                wordScores.put(word, wordScore);
            }
        }

        // Use backtracking to find the maximum score
        return MaximumScoreWordsFormedByLettersHelper(validWords, wordScores, letterCount, 0);
    }
    //endregion

    //region Backtracking
    public static List<String> BinaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        if (root != null) {
            BinaryTreePathsHelper(root, "", paths);
        }
        return paths;
    }

    public static int LetterTilePossibilities(String tiles) {
        Set<String> resultSet = new HashSet<>();
        char[] chars = tiles.toCharArray();
        boolean[] used = new boolean[chars.length];
        LetterTilePossibilitiesHelper(chars, used, "", resultSet);
        return resultSet.size();
    }

    //Combination Iterator is added as a different DTO class

    public static List<String> LetterCasePermutation_2(String s){
        return LetterCasePermutation_1(s);
    }

    public static String TheKthLexicographicalStringOfAllHappyStringsOfLengthN(int n, int k) {
        List<String> happyStrings = new ArrayList<>();
        TheKthLexicographicalStringOfAllHappyStringsOfLengthNHelper(n, "", happyStrings);

        if (k <= happyStrings.size()) {
            return happyStrings.get(k - 1); // k is 1-indexed
        } else {
            return "";
        }
    }
    //endregion

    //region String Matching
    public static int CheckIfAWordOccursAsAPrefixOfAnyWordInASentence(String sentence, String searchWord) {
        // Split the sentence into words
        String[] words = sentence.split(" ");

        // Iterate through the words
        for (int i = 0; i < words.length; i++) {
            // Check if the current word starts with the searchWord
            if (words[i].startsWith(searchWord)) {
                // Return the 1-based index
                return i + 1;
            }
        }

        // If no match found, return -1
        return -1;
    }

    public static List<String> StringMatchingInAnArray(String[] words) {
        List<String> result = new ArrayList<>();

        // Compare each word with every other word
        for (int i = 0; i < words.length; i++) {
            for (int j = 0; j < words.length; j++) {
                // Check if words[i] is a substring of words[j]
                if (i != j && words[j].contains(words[i])) {
                    result.add(words[i]);
                    break; // No need to check further, we already found it
                }
            }
        }

        return result;
    }

    public static boolean RotateString(String s, String goal) {
        // Step 1: Check if the lengths are different
        if (s.length() != goal.length()) {
            return false;
        }

        // Step 2: Check if goal is a substring of s + s
        String doubled = s + s;
        return doubled.contains(goal);
    }

    public static boolean RepeatedSubstringPattern(String s) {
        // Concatenate s with itself
        String doubled = s + s;
        // Remove the first and last characters and check if s exists within the modified string
        return doubled.substring(1, doubled.length() - 1).contains(s);
    }

    public static int MaximumRepeatingSubstring(String sequence, String word){
        int k = 0;
        StringBuilder repeated = new StringBuilder();

        // Continue appending the word and checking if it's a substring of sequence
        while (sequence.contains(repeated.append(word).toString())) {
            k++;
        }

        return k;
    }
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

    private static String LongestNiceSubstringHelper(String s, int start, int end) {

        if (end - start < 2) {
            return ""; // A string of length 1 or less cannot be nice
        }

        Set<Character> set = new HashSet<>();
        for (int i = start; i < end; i++) {
            set.add(s.charAt(i));
        }

        for (int i = start; i < end; i++) {
            char c = s.charAt(i);
            // If the string contains either only uppercase or only lowercase for this character
            if (set.contains(Character.toUpperCase(c)) && set.contains(Character.toLowerCase(c))) {
                continue; // This character satisfies the nice condition
            }

            // Split the string at the invalid character
            String left = LongestNiceSubstringHelper(s, start, i);
            String right = LongestNiceSubstringHelper(s, i + 1, end);

            // Return the longer of the two, or the left if they're the same length
            return left.length() >= right.length() ? left : right;
        }

        // If we looped through the entire string and all characters are nice
        return s.substring(start, end);
    }

    private static int MaximizeTheConfusionOfAnExamHelper(String answerKey, int k, char ch) {
        int left = 0;
        int maxLen = 0;
        int count = 0; // To count the number of flips (non-ch characters in the window)

        for (int right = 0; right < answerKey.length(); right++) {
            // If the current character is not ch, we count it as a flip
            if (answerKey.charAt(right) != ch) {
                count++;
            }

            // If the number of flips exceeds k, shrink the window from the left
            while (count > k) {
                if (answerKey.charAt(left) != ch) {
                    count--;
                }
                left++;
            }

            // Update the maximum window size
            maxLen = Math.max(maxLen, right - left + 1);
        }

        return maxLen;
    }

    private static void LetterCasePermutationHelper(char[] chars, int index, List<String> result) {
        if (index == chars.length) {
            // If we have processed all characters, add the current permutation to the result
            result.add(new String(chars));
            return;
        }

        // Backtrack without changing the current character (continue as is)
        LetterCasePermutationHelper(chars, index + 1, result);

        // If the current character is a letter, we toggle its case and backtrack again
        if (Character.isLetter(chars[index])) {
            // Change case: toggle between uppercase and lowercase
            chars[index] = Character.isLowerCase(chars[index])
                    ? Character.toUpperCase(chars[index])
                    : Character.toLowerCase(chars[index]);

            // Backtrack with the toggled character
            LetterCasePermutationHelper(chars, index + 1, result);

            // Revert the case change to continue with other possibilities
            chars[index] = Character.isLowerCase(chars[index])
                    ? Character.toUpperCase(chars[index])
                    : Character.toLowerCase(chars[index]);
        }
    }

    private static void GenerateParenthesesHelper(List<String> result, String current, int open, int close, int max) {
        // Base case: if the current string has 2 * n characters, it is a valid combination
        if (current.length() == max * 2) {
            result.add(current);
            return;
        }

        // Add an opening parenthesis if possible
        if (open < max) {
            GenerateParenthesesHelper(result, current + "(", open + 1, close, max);
        }

        // Add a closing parenthesis if it will not invalidate the combination
        if (close < open) {
            GenerateParenthesesHelper(result, current + ")", open, close + 1, max);
        }
    }

    private static final Map<String, List<Integer>> memo = new HashMap<>();

    private static void PalindromePartitioningHelper(String s, int start, List<String> currentPartition, List<List<String>> result) {
        if (start == s.length()) {
            result.add(new ArrayList<>(currentPartition));
            return;
        }

        for (int end = start + 1; end <= s.length(); end++) {
            String substring = s.substring(start, end);
            if (IsPalindrome(substring)) {
                currentPartition.add(substring);
                PalindromePartitioningHelper(s, end, currentPartition, result);
                currentPartition.remove(currentPartition.size() - 1);
            }
        }
    }

    private static boolean IsPalindrome(String s) {
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    private static char FindKthBitInNthBinaryStringHelper(int n, int k) {
        int len = (1 << n) - 1; // Length of Sn, which is 2^n - 1

        // Base case
        if (n == 1) {
            return '0'; // S1 = "0"
        }

        int mid = len / 2 + 1;

        if (k == mid) {
            return '1'; // Middle of Sn is always '1'
        } else if (k < mid) {
            // k-th bit is in the first half, which is S_{n-1}
            return FindKthBitInNthBinaryStringHelper(n - 1, k);
        } else {
            // k-th bit is in the second half, which is the reversed inverted S_{n-1}
            char result = FindKthBitInNthBinaryStringHelper(n - 1, len - k + 1);
            return result == '0' ? '1' : '0'; // Invert the result
        }
    }

    private static int MaximumScoreWordsFormedByLettersHelper(List<String> words, Map<String, Integer> wordScores, Map<Character, Integer> letterCount, int index) {
        if (index == words.size()) {
            return 0;
        }

        // Exclude current word
        int maxScore = MaximumScoreWordsFormedByLettersHelper(words, wordScores, letterCount, index + 1);

        // Include current word if it can be formed
        String currentWord = words.get(index);
        if (CanFormWord(currentWord, letterCount)) {
            // Update letter counts
            Map<Character, Integer> newLetterCount = new HashMap<>(letterCount);
            for (char c : currentWord.toCharArray()) {
                newLetterCount.put(c, newLetterCount.get(c) - 1);
            }

            // Calculate score including this word
            int includeScore = wordScores.get(currentWord) + MaximumScoreWordsFormedByLettersHelper(words, wordScores, newLetterCount, index + 1);
            maxScore = Math.max(maxScore, includeScore);
        }

        return maxScore;
    }

    private static boolean CanFormWord(String word, Map<Character, Integer> letterCount) {
        Map<Character, Integer> wordCount = new HashMap<>();
        for (char c : word.toCharArray()) {
            wordCount.put(c, wordCount.getOrDefault(c, 0) + 1);
        }
        for (Map.Entry<Character, Integer> entry : wordCount.entrySet()) {
            if (entry.getValue() > letterCount.getOrDefault(entry.getKey(), 0)) {
                return false;
            }
        }
        return true;
    }

    private static int GetWordScore(String word, int[] score) {
        int totalScore = 0;
        for (char c : word.toCharArray()) {
            totalScore += score[c - 'a'];
        }
        return totalScore;
    }

    private static void BinaryTreePathsHelper(TreeNode node, String path, List<String> paths) {
        if (node != null) {
            path += Integer.toString(node.val);
            // Check if it's a leaf node
            if (node.left == null && node.right == null) {
                paths.add(path); // Add the current path to the result
            } else {
                // Continue to explore left and right subtrees
                if (node.left != null) {
                    BinaryTreePathsHelper(node.left, path + "->", paths);
                }
                if (node.right != null) {
                    BinaryTreePathsHelper(node.right, path + "->", paths);
                }
            }
        }
    }

    private static void LetterTilePossibilitiesHelper(char[] chars, boolean[] used, String current, Set<String> resultSet) {
        for (int i = 0; i < chars.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            String newSequence = current + chars[i];
            resultSet.add(newSequence);
            LetterTilePossibilitiesHelper(chars, used, newSequence, resultSet);
            used[i] = false; // backtrack
        }
    }

    private static void TheKthLexicographicalStringOfAllHappyStringsOfLengthNHelper(int n, String current, List<String> result) {
        if (current.length() == n) {
            result.add(current);
            return;
        }

        for (char ch : new char[] {'a', 'b', 'c'}) {
            if (current.isEmpty() || current.charAt(current.length() - 1) != ch) {
                TheKthLexicographicalStringOfAllHappyStringsOfLengthNHelper(n, current + ch, result);
            }
        }
    }
    //endregion
}
