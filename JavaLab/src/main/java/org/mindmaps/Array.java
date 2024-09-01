package org.mindmaps;

import org.dto.Pair;

import java.util.*;

public class Array {
    //region Hashing
    public static int NumberOfGoodPairs(int[] nums) {
        int n = nums.length;
        int countOfGoodPair = 0;
        for(int i = 0; i < n; i++) {
            for(int j = i + 1; j < n; j++) {
                if(nums[i] == nums[j] && i < j) {
                    countOfGoodPair++;
                }
            }
        }

        return countOfGoodPair;
    }

    public static int CountNumberOfPairsWithAbsoluteDifferenceK(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i < j && Math.abs(nums[i] - nums[j]) == k) {
                    count++;
                }
            }
        }

        return count;
    }

    public static int CountTheNumberOfConsistentStrings(String allowed, String[] words) {
        if(allowed.isEmpty() || words.length == 0) return 0;
        List<Character> allowedCharsList = new ArrayList<>();
        int count = words.length;

        for(char a : allowed.toCharArray()) {
            allowedCharsList.add(a);
        }

        for(String word : words) {
            for(char a : word.toCharArray()) {
                if(!allowedCharsList.contains(a)){
                    count--;
                    break;
                }
            }
        }
        return count;
    }

    public static int UniqueMorseCodeWords(String[] words) {
        String[] morseCodes = new String[] {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};

        //Prepare dictionary
        Map<Character, String> morseDictionary = new HashMap<>();
        char a = 'a';
        for(String code : morseCodes) {
            morseDictionary.put(a,code);
            a++;
        }

        //Prepare MorseCode for the given words
        Map<String, String> morseCodeOfWords = new HashMap<>();
        for(String word : words) {
            String morseCodeWord = "";
            for (char c : word.toCharArray()) {
                morseCodeWord += morseDictionary.get(c);
            }
            morseCodeOfWords.put(word,morseCodeWord);
        }

        //Find the equal codes
        Map<String, Integer> countOfMorseCode = new HashMap<>();
        for(Map.Entry<String, String> morseCodeOfWord : morseCodeOfWords.entrySet()) {
            String code = morseCodeOfWord.getValue();
            countOfMorseCode.put(code, countOfMorseCode.getOrDefault(code, 0) + 1);
        }

        //Get the max count
        return countOfMorseCode.size();
    }
    //endregion

    //region Bit Manipulation
    public static int[] DecodeXORedArray(int[] encoded, int first) {
        int n = encoded.length + 1;
        int[] arr = new int[n];
        arr[0] = first;

        for (int i = 0; i < encoded.length; i++) {
            arr[i + 1] = arr[i] ^ encoded[i];
        }

        return arr;
    }

    public static int SumOfAllSubsetXORTotals(int[] nums) {
        return subsetXORSumHelper(nums, 0, 0);
    }

    public static boolean DivideArrayIntoEqualPairs(int[] nums) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();

        // Count frequency of each element
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        // Check if each frequency is even
        for (int count : frequencyMap.values()) {
            if (count % 2 != 0) {
                return false;
            }
        }

        return true;

    }

    public static int[] SetMismatch(int[] nums) {
        int n = nums.length;
        int[] result = new int[2];
        Map<Integer, Integer> countMap = new HashMap<>();

        // Populate the frequency map
        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }

        // Identify the duplicated and missing numbers
        for (int i = 1; i <= n; i++) {
            int count = countMap.getOrDefault(i, 0);
            if (count == 2) {
                result[0] = i; // duplicated number
            } else if (count == 0) {
                result[1] = i; // missing number
            }
        }

        return result;
    }
    //endregion

    //region Two Pointer
    public static int[][] FlippingAnImage(int[][] image) {
        int n = image.length;

        for (int i = 0; i < n; i++) {
            // Flip and invert each row simultaneously
            for (int j = 0; j < (n + 1) / 2; j++) {
                // Swap the elements and invert them
                int temp = image[i][j] ^ 1;
                image[i][j] = image[i][n - 1 - j] ^ 1;
                image[i][n - 1 - j] = temp;
            }
        }

        return image;
    }

    public static String FindFirstPalindromicStringInTheArray(String[] words) {
        for (String word : words) {
            if (isPalindrome(word)) {
                return word;
            }
        }
        return "";
    }

    public static int[] DIStringMatch(String s) {
        int n = s.length();
        Stack<Integer> stack = new Stack<>();
        List<Integer> result = new ArrayList<>();

        // Traverse the string and push numbers onto the stack
        for (int i = 0; i <= n; i++) {
            stack.push(i); // Push the current number

            // If we encounter 'I' or reach the end, pop all elements from the stack
            if (i == n || s.charAt(i) == 'I') {
                while (!stack.isEmpty()) {
                    result.add(stack.pop());
                }
            }
        }

        // Convert List<Integer> to int[]
        return result.stream().mapToInt(i -> i).toArray();
    }
    //endregion

    //region Privates
    private static int subsetXORSumHelper(int[] nums, int index, int currentXOR) {
        if (index == nums.length) {
            return currentXOR;
        }

        // Include nums[index] in the subset
        int withCurrent = subsetXORSumHelper(nums, index + 1, currentXOR ^ nums[index]);

        // Exclude nums[index] from the subset
        int withoutCurrent = subsetXORSumHelper(nums, index + 1, currentXOR);

        return withCurrent + withoutCurrent;
    }

    private static boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    //endregion
}
