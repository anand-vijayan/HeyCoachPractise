package org.dto;

import java.util.HashMap;

class LRUCache {
    private HashMap<Integer, Node> map; // HashMap to store key-node pairs
    private Node head; // Most recently used node
    private Node tail; // Least recently used node
    private int capacity; // Maximum capacity of the cache
    private int size;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        this.map = new HashMap<>();
        this.head = new Node(0, 0); // Dummy head
        this.tail = new Node(0, 0); // Dummy tail
        head.next = tail; // Link head and tail
        tail.prev = head;
    }

    // Get value from the cache
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1; // Key not found
        }
        Node node = map.get(key);
        moveToFront(node); // Move accessed node to the front
        return node.value; // Return the value
    }

    // Set key-value in the cache
    public void set(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value; // Update value
            moveToFront(node); // Move to front
        } else {
            if (size == capacity) {
                removeLRU(); // Remove least recently used item
            }
            Node newNode = new Node(key, value);
            addToFront(newNode); // Add new node to the front
            map.put(key, newNode); // Update map
        }
    }

    // Move the given node to the front of the linked list
    private void moveToFront(Node node) {
        removeNode(node);
        addToFront(node);
    }

    // Add a node right after the head
    private void addToFront(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
        size++;
    }

    // Remove a node from the linked list
    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        size--;
    }

    // Remove the least recently used node
    private void removeLRU() {
        if (tail.prev == head) return; // No node to remove
        Node lruNode = tail.prev; // Get the LRU node
        removeNode(lruNode); // Remove from linked list
        map.remove(lruNode.key); // Remove from hashmap
    }
}
