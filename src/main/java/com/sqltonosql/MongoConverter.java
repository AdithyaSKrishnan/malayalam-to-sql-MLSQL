package com.sqltonosql;

import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.expression.operators.relational.EqualsTo;
import net.sf.jsqlparser.expression.operators.relational.GreaterThan;
import net.sf.jsqlparser.expression.operators.relational.MinorThan;
import net.sf.jsqlparser.expression.operators.conditional.AndExpression;
import net.sf.jsqlparser.expression.operators.conditional.OrExpression;
import net.sf.jsqlparser.expression.operators.relational.LikeExpression;
import net.sf.jsqlparser.expression.StringValue;
import net.sf.jsqlparser.expression.Parenthesis;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.delete.Delete;
import net.sf.jsqlparser.statement.insert.Insert;
import net.sf.jsqlparser.statement.update.Update;
import net.sf.jsqlparser.statement.update.UpdateSet;
import net.sf.jsqlparser.statement.select.Join;
import net.sf.jsqlparser.expression.operators.relational.InExpression;
import net.sf.jsqlparser.expression.operators.relational.ExpressionList;
import net.sf.jsqlparser.statement.select.ParenthesedSelect;
import net.sf.jsqlparser.expression.operators.relational.ExpressionList;

import java.util.ArrayList;
import java.util.List;

public class MongoConverter {

    public String convert(MongoQueryModel model) {
        boolean isAggregation = (model.getJoins() != null && !model.getJoins().isEmpty()) ||
                (model.getGroupByFields() != null && !model.getGroupByFields().isEmpty());

        if (isAggregation) {
            return convertAggregation(model);
        } else {
            return convertFind(model);
        }
    }

    // --- Select / Find Conversion ---

    private String convertFind(MongoQueryModel model) {
        StringBuilder mongoQuery = new StringBuilder();
        mongoQuery.append("db.");
        mongoQuery.append(model.getCollectionName());
        mongoQuery.append(".find(");

        // 1. Filter (Query)
        String filterJson = convertFilter(model.getFilterExpression());
        mongoQuery.append(filterJson);

        // 2. Projection
        String projectionJson = convertProjection(model.getProjectionFields(), model.getCollectionName());
        if (!projectionJson.equals("{}")) {
            mongoQuery.append(", ").append(projectionJson);
        }

        mongoQuery.append(")");

        // 3. Sort
        if (model.getSortFields() != null && !model.getSortFields().isEmpty()) {
            StringBuilder sortJson = new StringBuilder("{");
            int i = 0;
            for (java.util.Map.Entry<String, Boolean> entry : model.getSortFields().entrySet()) {
                String field = resolveField(entry.getKey(), model.getCollectionName());
                int dir = entry.getValue() ? 1 : -1;
                sortJson.append("\"").append(field).append("\": ").append(dir);
                if (i < model.getSortFields().size() - 1)
                    sortJson.append(", ");
                i++;
            }
            sortJson.append("}");
            mongoQuery.append(".sort(").append(sortJson).append(")");
        }

        // 4. Limit
        if (model.getLimit() != null) {
            mongoQuery.append(".limit(").append(model.getLimit()).append(")");
        }

        return mongoQuery.toString();
    }

    private String convertAggregation(MongoQueryModel model) {
        StringBuilder pipeline = new StringBuilder();
        pipeline.append("db.").append(model.getCollectionName()).append(".aggregate([");

        // 1. Lookups (Joins)
        for (Join join : model.getJoins()) {
            pipeline.append(convertJoin(join)).append(", ");
        }

        // 2. Match (Filter)
        String filterJson = convertFilter(model.getFilterExpression());
        if (!filterJson.equals("{}")) {
            pipeline.append("{\"$match\": ").append(filterJson).append("}, ");
        }

        // 3. Group By (if exists) -> This replaces Projection usually
        if (model.getGroupByFields() != null && !model.getGroupByFields().isEmpty()) {
            StringBuilder groupJson = new StringBuilder("{\"$group\": {");

            // _id
            if (model.getGroupByFields().size() == 1) {
                String field = resolveField(model.getGroupByFields().get(0), model.getCollectionName());
                groupJson.append("\"_id\": \"$").append(field).append("\"");
            } else {
                groupJson.append("\"_id\": {");
                for (int i = 0; i < model.getGroupByFields().size(); i++) {
                    String field = resolveField(model.getGroupByFields().get(i), model.getCollectionName());
                    groupJson.append("\"").append(field).append("\": \"$").append(field).append("\"");
                    if (i < model.getGroupByFields().size() - 1)
                        groupJson.append(", ");
                }
                groupJson.append("}");
            }

            // Accumulators from Projection (e.g. COUNT(*))
            for (String proj : model.getProjectionFields()) {
                String upper = proj.toUpperCase();
                if (upper.contains("COUNT(*)") || upper.contains("COUNT(1)")) {
                    groupJson.append(", \"count\": {\"$sum\": 1}");
                } else if (upper.startsWith("SUM(") && upper.endsWith(")")) {
                    String field = proj.substring(4, proj.length() - 1); // Extract age from SUM(age)
                    groupJson.append(", \"sum_").append(field).append("\": {\"$sum\": \"$").append(field).append("\"}");
                }
            }

            groupJson.append("}}");
            pipeline.append(groupJson);
        } else {
            // 3b. Projection (Only if NO group by)
            String projectionJson = convertProjection(model.getProjectionFields(), model.getCollectionName());
            if (!projectionJson.equals("{}")) {
                pipeline.append("{\"$project\": ").append(projectionJson).append("}");
            } else {
                // Remove trailing comma if no projection
                if (pipeline.lastIndexOf(", ") == pipeline.length() - 2) {
                    pipeline.setLength(pipeline.length() - 2);
                }
            }
        }

        // 4. Sort
        if (model.getSortFields() != null && !model.getSortFields().isEmpty()) {
            pipeline.append(", {\"$sort\": {");
            int i = 0;
            for (java.util.Map.Entry<String, Boolean> entry : model.getSortFields().entrySet()) {
                String field = resolveField(entry.getKey(), model.getCollectionName());
                int dir = entry.getValue() ? 1 : -1;
                pipeline.append("\"").append(field).append("\": ").append(dir);
                if (i < model.getSortFields().size() - 1)
                    pipeline.append(", ");
                i++;
            }
            pipeline.append("}}");
        }

        // 5. Limit
        if (model.getLimit() != null) {
            pipeline.append(", {\"$limit\": ").append(model.getLimit()).append("}");
        }

        pipeline.append("])");
        return pipeline.toString();
    }

    // --- CRUD Operations ---

    public String convertInsert(Insert insert) {
        String collection = insert.getTable().getName();
        StringBuilder json = new StringBuilder("db.").append(collection).append(".insertOne({");

        List<net.sf.jsqlparser.schema.Column> columns = insert.getColumns();
        // Since JSqlParser 4.6+, values are ExpressionList or Select
        // We assume simple VALUES (...)
        if (insert.getValues() != null && insert.getValues().getExpressions() != null) {
            List<?> values = insert.getValues().getExpressions();
            for (int i = 0; i < columns.size(); i++) {
                String col = columns.get(i).getColumnName();
                String val = values.get(i).toString();
                json.append("\"").append(col).append("\": ").append(val);
                if (i < columns.size() - 1)
                    json.append(", ");
            }
        }

        json.append("})");
        return json.toString();
    }

    public String convertUpdate(Update update) {
        String collection = update.getTable().getName();
        StringBuilder json = new StringBuilder("db.").append(collection).append(".updateMany(");

        // 1. Filter
        if (update.getWhere() != null) {
            String filter = convertFilter(update.getWhere());
            json.append(filter).append(", ");
        } else {
            json.append("{}, ");
        }

        // 2. Set
        json.append("{\"$set\": {");
        List<UpdateSet> sets = update.getUpdateSets();
        for (int i = 0; i < sets.size(); i++) {
            UpdateSet set = sets.get(i);
            String col = set.getColumns().get(0).getColumnName();
            String val = set.getValues().get(0).toString();
            json.append("\"").append(col).append("\": ").append(val);
            if (i < sets.size() - 1)
                json.append(", ");
        }
        json.append("}})");

        return json.toString();
    }

    public String convertDelete(Delete delete) {
        String collection = delete.getTable().getName();
        StringBuilder json = new StringBuilder("db.").append(collection).append(".deleteMany(");
        String filter = convertFilter(delete.getWhere());
        json.append(filter).append(")");
        return json.toString();
    }

    // --- Helpers ---

    private String convertJoin(Join join) {
        final String[] fields = new String[2];
        String rightTable = ((Table) join.getRightItem()).getName();
        if (join.getOnExpression() instanceof EqualsTo) {
            EqualsTo eq = (EqualsTo) join.getOnExpression();
            fields[0] = eq.getLeftExpression().toString();
            fields[1] = eq.getRightExpression().toString();
        }
        String localField = stripTable(fields[0]);
        String foreignField = stripTable(fields[1]);

        return String.format(
                "{\"$lookup\": {\"from\": \"%s\", \"localField\": \"%s\", \"foreignField\": \"%s\", \"as\": \"%s\"}}",
                rightTable, localField, foreignField, rightTable);
    }

    private String stripTable(String raw) {
        if (raw == null)
            return "";
        if (raw.contains(".")) {
            return raw.split("\\.")[1];
        }
        return raw;
    }

    private String resolveField(String raw, String primaryCollection) {
        if (raw == null)
            return "";
        if (raw.startsWith(primaryCollection + ".")) {
            return raw.substring(primaryCollection.length() + 1);
        }
        return raw;
    }

    private String convertFilter(Expression filterExpression) {
        if (filterExpression == null) {
            return "{}";
        }
        try {
            String json = parseExpressionRecursive(filterExpression);
            return "{" + json + "}";
        } catch (Exception e) {
            return "{\"_error\": \"Unsupported Expression: " + e.getMessage() + "\"}";
        }
    }

    private String parseExpressionRecursive(Expression expr) {
        if (expr instanceof GreaterThan) {
            GreaterThan gt = (GreaterThan) expr;
            String left = stripTable(gt.getLeftExpression().toString());
            String right = gt.getRightExpression().toString();
            return String.format("\"%s\": {\"$gt\": %s}", left, right);
        } else if (expr instanceof MinorThan) {
            MinorThan lt = (MinorThan) expr;
            String left = stripTable(lt.getLeftExpression().toString());
            String right = lt.getRightExpression().toString();
            return String.format("\"%s\": {\"$lt\": %s}", left, right);
        } else if (expr instanceof EqualsTo) {
            EqualsTo eq = (EqualsTo) expr;
            String left = stripTable(eq.getLeftExpression().toString());
            String right = eq.getRightExpression().toString();
            return String.format("\"%s\": %s", left, right);
        } else if (expr instanceof AndExpression) {
            AndExpression and = (AndExpression) expr;
            String leftJson = parseExpressionRecursive(and.getLeftExpression());
            String rightJson = parseExpressionRecursive(and.getRightExpression());
            return leftJson + ", " + rightJson;
        } else if (expr instanceof OrExpression) {
            OrExpression or = (OrExpression) expr;
            String leftJson = parseExpressionRecursive(or.getLeftExpression());
            String rightJson = parseExpressionRecursive(or.getRightExpression());
            return String.format("\"$or\": [ {%s}, {%s} ]", leftJson, rightJson);
        } else if (expr instanceof LikeExpression) {
            LikeExpression like = (LikeExpression) expr;
            String left = stripTable(like.getLeftExpression().toString());
            String right = like.getRightExpression().toString();
            String pattern = right.replace("'", "");
            if (pattern.startsWith("%") && pattern.endsWith("%")) {
                pattern = pattern.substring(1, pattern.length() - 1);
            } else if (pattern.startsWith("%")) {
                pattern = pattern.substring(1) + "$";
            } else if (pattern.endsWith("%")) {
                pattern = "^" + pattern.substring(0, pattern.length() - 1);
            }
            return String.format("\"%s\": {\"$regex\": \"%s\"}", left, pattern);
        } else if (expr instanceof Parenthesis) {
            Parenthesis parenthesis = (Parenthesis) expr;
            return parseExpressionRecursive(parenthesis.getExpression());
        } else if (expr instanceof InExpression) {
            InExpression inExpr = (InExpression) expr;
            String left = stripTable(inExpr.getLeftExpression().toString());
            String operator = inExpr.isNot() ? "$nin" : "$in";

            Expression rightExpr = inExpr.getRightExpression();
            if (rightExpr instanceof ExpressionList) {
                ExpressionList exprList = (ExpressionList) rightExpr;
                List<String> values = new ArrayList<>();
                for (Object item : exprList.getExpressions()) {
                    if (item instanceof Expression) {
                        values.add(((Expression) item).toString());
                    } else {
                        values.add(item.toString());
                    }
                }
                return String.format("\"%s\": {\"%s\": [%s]}", left, operator, String.join(", ", values));
            } else if (rightExpr instanceof Parenthesis) {
                // Sometimes IN (a,b) is parsed as IN (ExpressionList) enclosed in Parenthesis
                Expression inner = ((Parenthesis) rightExpr).getExpression();
                if (inner instanceof ExpressionList) {
                    ExpressionList exprList = (ExpressionList) inner;
                    List<String> values = new ArrayList<>();
                    for (Object item : exprList.getExpressions()) {
                        if (item instanceof Expression) {
                            values.add(((Expression) item).toString());
                        } else {
                            values.add(item.toString());
                        }
                    }
                    return String.format("\"%s\": {\"%s\": [%s]}", left, operator, String.join(", ", values));
                }
            } else if (rightExpr instanceof ParenthesedSelect) {
                // Handle subquery e.g. IN (SELECT role FROM department)
                ParenthesedSelect subSelect = (ParenthesedSelect) rightExpr;
                return String.format("\"%s\": {\"%s\": \"<subquery: %s>\"}", left, operator,
                        subSelect.toString().replace("\"", "\\\""));
            }
            throw new IllegalArgumentException(
                    "Unsupported right expression for IN expression: " + rightExpr.getClass().getName());
        }
        throw new IllegalArgumentException(expr.getClass().getSimpleName());
    }

    private String convertProjection(List<String> fields, String primaryCollection) {
        if (fields == null || fields.isEmpty() || (fields.size() == 1 && fields.get(0).equals("*"))) {
            return "{}";
        }

        StringBuilder json = new StringBuilder("{");
        for (int i = 0; i < fields.size(); i++) {
            json.append("\"").append(resolveField(fields.get(i), primaryCollection)).append("\": 1");
            if (i < fields.size() - 1) {
                json.append(", ");
            }
        }
        json.append("}");
        return json.toString();
    }
}
