package com.sqltonosql;

import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.statement.select.Join;
import java.util.ArrayList;
import java.util.List;

public class MongoQueryModel {
    private String collectionName;
    private List<String> projectionFields = new ArrayList<>();
    private Expression filterExpression;
    private List<Join> joins = new ArrayList<>();
    private java.util.Map<String, Boolean> sortFields = new java.util.LinkedHashMap<>(); // Field -> isAscending
    private Long limit = null;
    private List<String> groupByFields = new ArrayList<>();

    public List<String> getGroupByFields() {
        return groupByFields;
    }

    public void addGroupByField(String field) {
        this.groupByFields.add(field);
    }

    public java.util.Map<String, Boolean> getSortFields() {
        return sortFields;
    }

    public void addSortField(String field, boolean isAsc) {
        this.sortFields.put(field, isAsc);
    }

    public Long getLimit() {
        return limit;
    }

    public void setLimit(Long limit) {
        this.limit = limit;
    }

    public List<Join> getJoins() {
        return joins;
    }

    public void addJoin(Join join) {
        this.joins.add(join);
    }
    // Optional schema reference if we want to carry it through
    // private Schema schema;

    public String getCollectionName() {
        return collectionName;
    }

    public void setCollectionName(String collectionName) {
        this.collectionName = collectionName;
    }

    public List<String> getProjectionFields() {
        return projectionFields;
    }

    public void addProjectionField(String field) {
        this.projectionFields.add(field);
    }

    public Expression getFilterExpression() {
        return filterExpression;
    }

    public void setFilterExpression(Expression filterExpression) {
        this.filterExpression = filterExpression;
    }

    @Override
    public String toString() {
        return "MongoQueryModel {" +
                "\n  collectionName='" + collectionName + '\'' +
                ",\n  projectionFields=" + projectionFields +
                ",\n  filterExpression=" + (filterExpression != null ? filterExpression.toString() : "null") +
                "\n}";
    }
}
