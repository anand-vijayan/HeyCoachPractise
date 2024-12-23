package org.dto;

import java.util.List;

public class Node {
    public int data;
    public int key;
    public int value;
    public Node next;
    public Node prev;
    public Node left;
    public Node right;
    public List<Node> children;

    public Node(int data) {
        this.data = data;
        this.next = null;
    }

    public Node(int key, int value) {
        this.key = key;
        this.value = value;
    }

    public Node(int _val, List<Node> _children) {
        value = _val;
        children = _children;
    }
}

