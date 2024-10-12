package org.mindmaps;

import java.util.*;

public class Maths {

    //region Number Theory
    public static int FindGreatestCommonDivisorOfArray(int[] nums){
        // Find the smallest and largest numbers in the array
        int minNum = Arrays.stream(nums).min().isPresent() ? Arrays.stream(nums).min().getAsInt() : Integer.MIN_VALUE;
        int maxNum = Arrays.stream(nums).max().isPresent() ? Arrays.stream(nums).max().getAsInt() : Integer.MAX_VALUE;

        // Compute and return the GCD of the smallest and largest numbers
        return gcd(minNum, maxNum);
    }

    public static int AddDigits(int num) {
        if (num == 0) {
            return 0;
        }

        //This formulae is called "Digital Root"
        return 1 + (num - 1) % 9;
    }

    public static boolean XOfAKindInADeckOfCards(int[] deck) {
        // Count the frequency of each card in the deck
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int card : deck) {
            countMap.put(card, countMap.getOrDefault(card, 0) + 1);
        }

        // Get the GCD of all frequencies
        int gcdValue = -1;
        for (int count : countMap.values()) {
            if (gcdValue == -1) {
                gcdValue = count;
            } else {
                gcdValue = gcd(gcdValue, count);
            }
        }

        // Return true if GCD is greater than 1, false otherwise
        return gcdValue >= 2;
    }

    public static List<String> SimplifiedFractions(int n) {
        List<String> result = new ArrayList<>();

        // Iterate over all possible denominators from 2 to n
        for (int denominator = 2; denominator <= n; denominator++) {
            // For each denominator, iterate over all possible numerators
            for (int numerator = 1; numerator < denominator; numerator++) {
                // Check if gcd(numerator, denominator) is 1 (i.e., the fraction is simplified)
                if (gcd(numerator, denominator) == 1) {
                    result.add(numerator + "/" + denominator);
                }
            }
        }

        return result;
    }

    public static long NumberOfPairsOfInterchangeableRectangles(int[][] rectangles){
        Map<String, Integer> ratioCount = new HashMap<>();
        long interchangeablePairs = 0;

        // Iterate through each rectangle
        for (int[] rectangle : rectangles) {
            int width = rectangle[0];
            int height = rectangle[1];

            // Compute the GCD of width and height to simplify the ratio
            int gcd = gcd(width, height);

            // Create a string representing the simplified ratio (width/gcd, height/gcd)
            String ratio = (width / gcd) + ":" + (height / gcd);

            // Count the number of occurrences of each ratio
            int count = ratioCount.getOrDefault(ratio, 0);

            // Each existing rectangle with the same ratio forms a pair with the current rectangle
            interchangeablePairs += count;

            // Update the count of the ratio in the map
            ratioCount.put(ratio, count + 1);
        }

        return interchangeablePairs;
    }
    //endregion

    //region Game Theory
    public static boolean DivisorGame(int n) {
        return n % 2 == 0;
    }

    public static boolean NimGame(int n) {
        return n % 4 != 0;
    }

    public static int MaximumNumberOfCoinsYouCanGet(int[] piles) {
        // Sort the piles in descending order
        Arrays.sort(piles);
        int n = piles.length / 3;
        int maxCoins = 0;

        // We start from the second largest in each triplet and skip the last (Bob's pick)
        for (int i = piles.length - 2; i >= n; i -= 2) {
            maxCoins += piles[i];
        }

        return maxCoins;
    }

    public static boolean StoneGame_1(int[] piles) {
        int n = piles.length;
        int[][] dp = new int[n][n];

        // Base case: when i == j, there's only one pile left, so the current player takes it
        for (int i = 0; i < n; i++) {
            dp[i][i] = piles[i];
        }

        // Fill the dp table
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i <= n - length; i++) {
                int j = i + length - 1;
                dp[i][j] = Math.max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1]);
            }
        }

        // If Alice can collect more stones than Bob, dp[0][n-1] > 0
        return dp[0][n-1] > 0;
    }

    public static int StoneGame_2(int[] piles) {
        int n = piles.length;
        // Suffix sum array to store the total number of stones from pile i to the end
        int[] suffixSum = new int[n];
        suffixSum[n - 1] = piles[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            suffixSum[i] = piles[i] + suffixSum[i + 1];
        }

        // DP table to store the result for dp[i][M]
        int[][] dp = new int[n][n + 1]; // M can be as large as n (worst case)

        // Base case: when we reach the end of the piles
        for (int i = 0; i < n; i++) {
            Arrays.fill(dp[i], 0);
        }

        // Fill DP table
        for (int i = n - 1; i >= 0; i--) {
            for (int M = 1; M <= n; M++) {
                // Max piles we can take is 2 * M or the remaining number of piles
                for (int X = 1; X <= 2 * M && i + X <= n; X++) {
                    // Current player takes all the stones remaining if possible
                    dp[i][M] = Math.max(dp[i][M], suffixSum[i] - (i + X < n ? dp[i + X][Math.max(M, X)] : 0));
                }
            }
        }

        // Result is dp[0][1] since we start with the first pile and M = 1
        return dp[0][1];
    }

    //endregion

    //region Geometry
    public static int MinimumTimeVisitingAllPoints(int[][] points) {
        int totalTime = 0;

        for (int i = 1; i < points.length; i++) {
            int xDiff = Math.abs(points[i][0] - points[i - 1][0]);
            int yDiff = Math.abs(points[i][1] - points[i - 1][1]);
            totalTime += Math.max(xDiff, yDiff);
        }

        return totalTime;
    }

    public static int ProjectionAreaOf3DShapes(int[][] grid) {
        int n = grid.length;
        int xyArea = 0;  // Area for the xy-plane
        int yzArea = 0;  // Area for the yz-plane
        int zxArea = 0;  // Area for the zx-plane

        // To keep track of max heights for columns
        int[] maxColumnHeights = new int[n];

        // Iterate over the grid to calculate areas
        for (int[] integers : grid) {
            int maxRowHeight = 0;  // To keep track of max height in the current row
            for (int j = 0; j < n; j++) {
                // Calculate xy-plane area
                if (integers[j] > 0) {
                    xyArea++;
                }
                // Calculate max height for the current row
                maxRowHeight = Math.max(maxRowHeight, integers[j]);
                // Calculate max height for the current column
                maxColumnHeights[j] = Math.max(maxColumnHeights[j], integers[j]);
            }
            // Add the max height of the current row to the yz-area
            yzArea += maxRowHeight;
        }

        // Sum max heights for each column to calculate zx-area
        for (int height : maxColumnHeights) {
            zxArea += height;
        }

        // Total area is the sum of all three projections
        return xyArea + yzArea + zxArea;
    }

    public static int[][] MatrixCellsInDistanceOrder(int rows, int cols, int rCenter, int cCenter) {
        List<int[]> coordinates = new ArrayList<>();

        // Step 1: Generate all coordinates
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                coordinates.add(new int[]{r, c});
            }
        }

        // Step 2: Sort coordinates by distance to (rCenter, cCenter)
        coordinates.sort(Comparator.comparingInt(coord ->
                Math.abs(coord[0] - rCenter) + Math.abs(coord[1] - cCenter)
        ));

        // Step 3: Convert list back to 2D array
        int[][] result = new int[rows * cols][2];
        for (int i = 0; i < coordinates.size(); i++) {
            result[i] = coordinates.get(i);
        }

        return result;
    }

    public static boolean CheckIfItIsAStraightLine(int[][] coordinates) {
        return Array.CheckIfItIsAStraightLine(coordinates);
    }

    public static boolean ValidBoomerang(int[][] points) {
        // Check for distinct points
        if (points[0][0] == points[1][0] && points[0][1] == points[1][1]) return false;
        if (points[0][0] == points[2][0] && points[0][1] == points[2][1]) return false;
        if (points[1][0] == points[2][0] && points[1][1] == points[2][1]) return false;

        // Check for collinearity using the area formula
        return (points[0][0] * (points[1][1] - points[2][1]) +
                points[1][0] * (points[2][1] - points[0][1]) +
                points[2][0] * (points[0][1] - points[1][1])) != 0;
    }
    //endregion

    //region Private Methods
    private static int gcd(int a, int b) {
        // Helper method to compute GCD using the Euclidean algorithm
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    //endregion
}
