package org.heycoach;

import java.util.ArrayList;
import java.util.List;

import static org.helpers.Common.*;
import static org.modules.BasicDataStructures.*;

public class Main {
    public static void main(String[] args)  {
        List<Integer> arr = new ArrayList<>();
        arr.add(1);
        arr.add(2);
        arr.add(-2);
        arr.add(3);
        PrintArray(RohanLovesZero2(arr));
    }
}

