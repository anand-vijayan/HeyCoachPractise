package org.heycoach;

import static org.helpers.Common.*;
import static org.mindmaps.Array.*;

public class Main {
    public static void main(String[] args)  {
        int[][] grid = new int[][] {{4,3,2,-1}, {3,2,1,-1},{1,1,-1,-2},{-1,-1,-2,-3}};
        //PrintArray(FindTargetIndicesAfterSortingArray(nums,target));
        System.out.println(CountNegativeNumbersInASortedMatrix(grid));
    }
}

