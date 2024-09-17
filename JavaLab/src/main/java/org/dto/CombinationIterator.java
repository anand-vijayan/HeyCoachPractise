package org.dto;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

class CombinationIterator {

    Iterator<String> iterator;

    public CombinationIterator(String characters, int combinationLength) {
        List<String> combinations = new ArrayList<>();
        makeCombinations(0, characters, new StringBuilder(), combinationLength, combinations, new boolean[26]);
        iterator = combinations.iterator();
    }

    public void makeCombinations(int start, String characters, StringBuilder curr, int maxLen, List<String> combinations, boolean[] seen) {
        if (curr.length() == maxLen) {
            combinations.add(curr.toString());
            return;
        }

        for(int i = start; i < characters.length(); i++) {
            curr.append(characters.charAt(i));
            makeCombinations(i+1, characters, curr, maxLen, combinations, seen);
            curr.deleteCharAt(curr.length() - 1);
        }
    }

    public String next() {
        return iterator.next();
    }

    public boolean hasNext() {
        return iterator.hasNext();
    }
}
