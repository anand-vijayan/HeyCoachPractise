package org.tgp;

import java.util.TreeSet;

public class Level_C4 {
    public static String RemoveKDigits(String num, int k) {
        StringBuilder result = new StringBuilder();
        int length = num.length();

        // Use a stack represented by a StringBuilder
        for (char digit : num.toCharArray()) {
            // Remove digits from the stack if the current digit is smaller
            // than the last digit in the stack (which we stored in result)
            while (!result.isEmpty() && k > 0 && result.charAt(result.length() - 1) > digit) {
                result.deleteCharAt(result.length() - 1); // Remove the last digit
                k--;
            }
            result.append(digit); // Add the current digit
        }

        // If k digits are still to be removed, remove from the end
        while (k > 0) {
            result.deleteCharAt(result.length() - 1);
            k--;
        }

        // Convert the result to string and remove leading zeros
        String finalResult = result.toString().replaceFirst("^0+", ""); // Remove leading zeros

        // Return "0" if the result is empty
        return finalResult.isEmpty() ? "0" : finalResult;
    }

    public static int CoachManish(int d, int n) {
        int maxGCD = 1; // Start with the smallest GCD possible

        // Iterate over all possible divisors of d
        for (int g = 1; g <= d; g++) {
            if (d % g == 0) { // g is a divisor of d
                // Check if g can be the GCD
                if (g * n <= d) {
                    maxGCD = Math.max(maxGCD, g); // Update the maximum GCD
                }
            }
        }

        return maxGCD;
    }

    public String RemoveDuplicateLetters(String s) {
        TreeSet<Character> uniqueLetters = new TreeSet<>();
        for(char a : s.toCharArray()) {
            uniqueLetters.add(a);
        }
        StringBuilder output = new StringBuilder();
        for(char a : uniqueLetters) {
            output.append(a);
        }
        return output.toString();
    }
}
