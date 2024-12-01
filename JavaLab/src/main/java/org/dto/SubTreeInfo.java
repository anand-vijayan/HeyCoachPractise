package org.dto;

public class SubTreeInfo {
    public boolean isBST;
    public int size;
    public int min;
    public int max;

    public SubTreeInfo(boolean isBST, int size, int min, int max) {
        this.isBST = isBST;
        this.size = size;
        this.min = min;
        this.max = max;
    }
}