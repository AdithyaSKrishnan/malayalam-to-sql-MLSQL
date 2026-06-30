package com.sqltonosql;

import java.util.List;
import java.util.Map;

public class Schema {
    // collection name -> info
    private Map<String, CollectionInfo> collections;

    public Map<String, CollectionInfo> getCollections() {
        return collections;
    }

    public void setCollections(Map<String, CollectionInfo> collections) {
        this.collections = collections;
    }

    public CollectionInfo getCollection(String name) {
        return collections != null ? collections.get(name) : null;
    }

    public static class CollectionInfo {
        private String type;
        private List<String> embeds;
        private List<String> references;

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public List<String> getEmbeds() {
            return embeds;
        }

        public void setEmbeds(List<String> embeds) {
            this.embeds = embeds;
        }

        public List<String> getReferences() {
            return references;
        }

        public void setReferences(List<String> references) {
            this.references = references;
        }

        @Override
        public String toString() {
            return "CollectionInfo{type='" + type + "', embeds=" + embeds + ", references=" + references + "}";
        }
    }

    @Override
    public String toString() {
        return "Schema{collections=" + collections + "}";
    }
}
