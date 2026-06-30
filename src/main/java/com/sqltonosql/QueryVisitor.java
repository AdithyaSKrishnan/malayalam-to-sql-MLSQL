package com.sqltonosql;

import net.sf.jsqlparser.statement.select.SelectVisitorAdapter;
import net.sf.jsqlparser.statement.select.PlainSelect;
import net.sf.jsqlparser.statement.select.SelectItem;
import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.schema.Table;

public class QueryVisitor extends SelectVisitorAdapter {
    private final MongoQueryModel queryModel;

    public QueryVisitor() {
        this.queryModel = new MongoQueryModel();
    }

    public MongoQueryModel getQueryModel() {
        return queryModel;
    }

    @Override
    public void visit(PlainSelect plainSelect) {
        // 1. Extract Table Name (Collection)
        if (plainSelect.getFromItem() instanceof Table) {
            Table table = (Table) plainSelect.getFromItem();
            queryModel.setCollectionName(table.getName());
        }

        // 1b. Extract Joins
        if (plainSelect.getJoins() != null) {
            for (net.sf.jsqlparser.statement.select.Join join : plainSelect.getJoins()) {
                queryModel.addJoin(join);
            }
        }

        // 2. Extract Projection Fields (SELECT columns)
        for (SelectItem<?> item : plainSelect.getSelectItems()) {
            queryModel.addProjectionField(item.toString());
        }

        // 3. Extract Filter (WHERE)
        if (plainSelect.getWhere() != null) {
            queryModel.setFilterExpression(plainSelect.getWhere());
        }

        // 4. Extract ORDER BY
        if (plainSelect.getOrderByElements() != null) {
            for (net.sf.jsqlparser.statement.select.OrderByElement orderBy : plainSelect.getOrderByElements()) {
                String field = orderBy.getExpression().toString();
                boolean isAsc = orderBy.isAsc();
                queryModel.addSortField(field, isAsc);
            }
        }

        // 5. Extract LIMIT
        if (plainSelect.getLimit() != null) {
            // JSqlParser Limit object has operations, but usually it's just a row count
            if (plainSelect.getLimit().getRowCount() != null) {
                try {
                    long limitVal = Long.parseLong(plainSelect.getLimit().getRowCount().toString());
                    queryModel.setLimit(limitVal);
                } catch (NumberFormatException e) {
                    // ignore complex limits for now
                }
            }
        }

        // 6. Extract GROUP BY
        if (plainSelect.getGroupBy() != null) {
            for (Object expr : plainSelect.getGroupBy().getGroupByExpressionList()) {
                queryModel.addGroupByField(expr.toString());
            }
        }
    }
}
