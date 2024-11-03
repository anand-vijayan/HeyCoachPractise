package org.dto;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class UnionFind {
    private Map<String, String> parent = new LinkedHashMap<>();

    public String find(String s) {
        if (!parent.containsKey(s)) parent.put(s, s);
        if (!parent.get(s).equals(s)) parent.put(s, find(parent.get(s)));
        return parent.get(s);
    }

    public void union(String s1, String s2) {
        String root1 = find(s1);
        String root2 = find(s2);
        if (!root1.equals(root2)) parent.put(root1, root2);
    }
}