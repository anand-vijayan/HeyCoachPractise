package org.dto;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class UnionFind {
    public Map<String, String> parent;
    public Map<String, Double> weight;

    public UnionFind() {
        parent = new HashMap<>();
        weight = new HashMap<>();
    }

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

    // Union two variables
    public void union(String x, String y, double value) {
        String rootX = find(x);
        String rootY = find(y);

        if (!rootX.equals(rootY)) {
            parent.put(rootX, rootY);
            weight.put(rootX, value * weight.get(y) / weight.get(x));
        }
    }

    // Add a variable and initialize its parent and weight
    public void add(String x) {
        if (!parent.containsKey(x)) {
            parent.put(x, x);
            weight.put(x, 1.0);
        }
    }
}