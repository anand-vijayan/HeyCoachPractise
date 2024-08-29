package org.heycoach;

import org.dto.*;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Vector;

import static org.helpers.Common.*;
import static org.helpers.Sorting.*;
import static org.modules.BasicDataStructures.*;

public class Main {
    public static void main(String[] args)  {
        Integer[] inArr = new Integer[] {1,2,3,4,5};
        Integer[] outArr = new Integer[] {4,3,2,1,5};
        System.out.print(CalculateTotalTrappedWorkers(new Vector<>(List.of(inArr)), new Vector<>(List.of(outArr))));
    }
}

