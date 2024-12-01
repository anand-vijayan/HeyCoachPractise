package org.heycoach;

import java.util.*;
import static org.modules.AdvancedDataStructure.*;

public class Main {
    public static void main(String[] args)  {

        int[] input = new int[] {5,15,1,3};

        for(int a : input) {
            System.out.print(MedianAtEveryStep(a) + " ");
        }
        System.out.println();
    }
}

