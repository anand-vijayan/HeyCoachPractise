package org.heycoach;

import java.util.*;

import static org.helpers.Common.*;
import static org.mindmaps.Array.*;
import static org.modules.DynamicProgramming.*;

public class Main {
    public static void main(String[] args)  {
        int[][] arr = new int[][] {{2,8,7},{7,1,3},{1,9,5}};
        System.out.println(RichestCustomerWealth(arr));
    }
}

