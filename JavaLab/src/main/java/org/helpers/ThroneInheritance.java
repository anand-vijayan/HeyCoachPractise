package org.helpers;

import java.util.*;

public class ThroneInheritance {

    // Tree of children for each person, and a set to track dead people
    private Map<String, List<String>> children;
    private Set<String> dead;
    private String king;

    public ThroneInheritance(String kingName) {
        this.king = kingName;
        this.children = new HashMap<>();
        this.dead = new HashSet<>();
        this.children.put(kingName, new ArrayList<>()); // King has no children initially
    }

    // Method to simulate birth event
    public void birth(String parentName, String childName) {
        // Add child to parent's list of children
        children.computeIfAbsent(parentName, k -> new ArrayList<>()).add(childName);
        // Ensure the child has an empty list of children initially
        children.putIfAbsent(childName, new ArrayList<>());
    }

    // Method to simulate death event
    public void death(String name) {
        dead.add(name); // Mark the person as dead
    }

    // Method to get the current inheritance order
    public List<String> getInheritanceOrder() {
        List<String> result = new ArrayList<>();
        dfs(king, result); // Start the DFS from the king
        return result;
    }

    // Depth First Search to traverse the family tree
    private void dfs(String person, List<String> result) {
        // If the person is not dead, add them to the inheritance order
        if (!dead.contains(person)) {
            result.add(person);
        }

        // Recurse for each child of the current person
        for (String child : children.get(person)) {
            dfs(child, result);
        }
    }
}
