package com.sqltonosql;

import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.Select;
import com.sqltonosql.Schema;
import com.sqltonosql.SchemaLoader;

public class Main {
    public static void main(String[] args) {
        java.util.Scanner scanner = new java.util.Scanner(System.in);
        System.out.println("SQL to NoSQL Conversion Tool");
        System.out.println("============================");

        try {
            // 1. Get Schema
            System.out.println("\nEnter Schema JSON (Press Enter to use default sample):");
            System.out.println("Tip: You can paste a minified JSON string.");
            String schemaJson = scanner.nextLine().trim();

            if (schemaJson.isEmpty()) {
                schemaJson = "{\n" +
                        "  \"student\": {\n" +
                        "    \"type\": \"collection\",\n" +
                        "    \"embeds\": [\"course\"],\n" +
                        "    \"references\": [\"department\"]\n" +
                        "  },\n" +
                        "  \"department\": {\n" +
                        "    \"type\": \"collection\"\n" +
                        "  }\n" +
                        "}";
                System.out.println("Using default schema.");
            }

            Schema schema = SchemaLoader.loadSchema(schemaJson);
            System.out.println("✅ Schema Loaded: " + schema.getCollections().keySet());

            // 2. Loop for SQL Input
            while (true) {
                System.out.println("\nEnter SQL Query (or type 'exit' to quit):");
                String sql = scanner.nextLine().trim();

                if (sql.equalsIgnoreCase("exit")) {
                    break;
                }

                if (sql.isEmpty()) {
                    continue;
                }

                try {
                    Statement statement = CCJSqlParserUtil.parse(sql);
                    MongoConverter converter = new MongoConverter();
                    String mongoQuery = "";

                    if (statement instanceof Select) {
                        Select select = (Select) statement;
                        QueryVisitor visitor = new QueryVisitor();
                        select.accept(visitor);
                        mongoQuery = converter.convert(visitor.getQueryModel());
                    } else if (statement instanceof net.sf.jsqlparser.statement.insert.Insert) {
                        mongoQuery = converter.convertInsert((net.sf.jsqlparser.statement.insert.Insert) statement);
                    } else if (statement instanceof net.sf.jsqlparser.statement.update.Update) {
                        mongoQuery = converter.convertUpdate((net.sf.jsqlparser.statement.update.Update) statement);
                    } else if (statement instanceof net.sf.jsqlparser.statement.delete.Delete) {
                        mongoQuery = converter.convertDelete((net.sf.jsqlparser.statement.delete.Delete) statement);
                    } else {
                        System.out.println("⚠️ Unsupported Statement Type.");
                        continue;
                    }

                    // Print Result
                    System.out.println("----- MongoDB Output -----");
                    System.out.println(mongoQuery);
                    System.out.println("--------------------------");

                } catch (Exception e) {
                    System.err.println("❌ Parsing failed: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("❌ Fatal Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            scanner.close();
            System.out.println("Exiting...");
        }
    }
}
