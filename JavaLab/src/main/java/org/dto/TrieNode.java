package org.dto;

import java.util.HashMap;
import java.util.Map;

public class TrieNode {
    public char data;
    public TrieNode[] children;
    public boolean isend;
    public TrieNode right;
    public TrieNode left;
    public long count;
    public Map<Character, TrieNode> childrenMap;
    public boolean isEndOfWord;

    public TrieNode() {
        childrenMap = new HashMap<>();
        isEndOfWord = false;
        children = new TrieNode[2];
    }

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
}
