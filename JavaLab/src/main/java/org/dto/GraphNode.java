package org.dto;

import java.util.ArrayList;
import java.util.List;

public class GraphNode {
    public int data;
    public List<GraphNode> neighbours;

    public GraphNode(int data) {
        this.data = data;
        this.neighbours = new ArrayList<>();
    }
}
