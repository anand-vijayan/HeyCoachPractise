package org.heycoach;

import java.util.*;

import static org.helpers.Common.*;
import static org.mindmaps.Array.*;
import static org.modules.DynamicProgramming.*;

public class Main {
    public static void main(String[] args)  {
        int[] cookies = new int[] {8,15,10,20,8};
        int k = 2;
        System.out.println(FairDistributionOfCookies(cookies,k));
    }
}

