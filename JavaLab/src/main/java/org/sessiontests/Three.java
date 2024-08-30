package org.sessiontests;

import java.util.*;

public class Three {
    public static int CountOfStudentsCannotEat(List<Integer> students, List<Integer> sandwiches) {
        Queue<Integer> studentsQueue = new LinkedList<>(students);
        int i = 0, count = 0;

        while(!studentsQueue.isEmpty() && count != studentsQueue.size()) {
            if(studentsQueue.peek() == sandwiches.get(i)) {
                studentsQueue.poll();
                i++;
                count = 0;
            } else {
                studentsQueue.offer(studentsQueue.poll());
                count++;
            }
        }

        return studentsQueue.size();
    }

    public static int FirstUniqueInteger(int[] v) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();

        // Count the frequency of each element
        for (int num : v) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        // Find the first element with frequency 1
        for (int num : v) {
            if (frequencyMap.get(num) == 1) {
                return num;
            }
        }

        // If there's no unique element, this part won't be reached
        // since the problem states the list is non-empty and there's a unique element
        return -1;
    }

    public static boolean AllAlphabetsOrNot(String str) {
        //First insert each unique characters to the collection.
        List<Character> charBucket = new ArrayList<>();
        for(char a : str.toCharArray()) {
            if(!charBucket.contains(Character.toLowerCase(a))) {
                charBucket.add(Character.toLowerCase(a));
            }
        }
        //Check the size of queue
        return charBucket.size() == 26;
    }
}
