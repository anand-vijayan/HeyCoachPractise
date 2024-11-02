package org.dto;

public class TrieNode {
    public char data;
    public TrieNode[] children;
    public boolean isend;
    public TrieNode right;
    public TrieNode left;
    public long count;

    public TrieNode(char data) {
        this.data = data;
        this.children = new TrieNode[26];
        this.isend = false;
        this.right = null;
        this.left = null;
        this.count = 0;

        for (int i = 0; i < 26; i++) {
            children[i] = null;
        }
    }

    public TrieNode() {
        // Each node can have 2 children: 0 or 1
        children = new TrieNode[2];
    }
}
