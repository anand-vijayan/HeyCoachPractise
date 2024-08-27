package org.heycoach;

import org.dto.*;

import static org.helpers.Common.*;
import static org.helpers.Sorting.*;
import static org.modules.BasicDataStructures.*;

public class Main {
    public static void main(String[] args)  {
        int[] nums1 = {0,0,1,2,0};
        PrintArray(BubbleSort(nums1));

        int[] nums2 = {2,0,1};
        PrintArray(BubbleSort(nums2));
    }
}

