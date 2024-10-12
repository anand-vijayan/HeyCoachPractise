package org.heycoach;

import java.util.ArrayList;
import java.util.List;

import static org.sessiontests.Five.SpecialArrangements;
import static org.tgp.Level_C3.*;

public class Main {
    public static void main(String[] args)  {
        List<Integer> nums = new ArrayList<>();
        nums.add(1);
        nums.add(2);
        nums.add(6);
        nums.add(5);
        nums.add(4);
        System.out.println(SpecialArrangements(nums));
    }
}

