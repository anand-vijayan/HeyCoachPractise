package org.dto;

public class Node {
    public int data;
    public int key;
    public int value;
    public Node next;
    public Node prev;

    public Node(int data) {
        this.data = data;
        this.next = null;
    }

    public Node(int key, int value) {
        this.key = key;
        this.value = value;
    }
}

