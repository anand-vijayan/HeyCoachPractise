package org.tgp;

public class Level_C2 {
    public static boolean ChocolateDistribution(int n) {
        return n % 2 == 0 && n > 2;
    }

    public static String WordCorrection(String s) {
        int upperCount = 0;
        int lowerCount = 0;

        // Count the number of uppercase and lowercase letters
        for (char c : s.toCharArray()) {
            if (Character.isUpperCase(c)) {
                upperCount++;
            } else {
                lowerCount++;
            }
        }

        // Compare counts and change the case accordingly
        return (upperCount > lowerCount) ? s.toUpperCase() : s.toLowerCase();
    }

    public static int BoardFilling(int n, int m) {
        return (m * n) / 2;
    }
}
