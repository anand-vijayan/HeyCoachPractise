package org.helpers;

import org.dto.TreeNode;

import java.util.HashSet;

public class FindElements {
    private HashSet<Integer> recoveredValues;

    public FindElements(TreeNode root) {
        recoveredValues = new HashSet<>();
        recover(root, 0);
    }

    public boolean find(int target) {
        return recoveredValues.contains(target);
    }

    private void recover(TreeNode node, int value) {
        if (node == null) return;

        // Recover current node's value
        node.val = value;
        recoveredValues.add(value);

        // Recursively recover left and right children
        if (node.left != null) recover(node.left, 2 * value + 1);
        if (node.right != null) recover(node.right, 2 * value + 2);
    }
}
